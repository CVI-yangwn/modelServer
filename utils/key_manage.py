import threading
import contextlib
import asyncio
import json
import os
import time
from datetime import datetime, date
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
from logger import set_logger

class NoAvailableKeyError(Exception):
    pass

class UserAccessDeniedError(Exception):
    pass


MODEL_LIMITS = {
    "gemini-2.5-pro": 200,
    "gemini-2.5-flash": 250,
    "gemini-2.0-flash": 200,
    "gemini-2.5-flash-lite": 1000,
    "default": 0
}


@dataclass
class UserAccessRecord:
    """用户访问记录"""
    username: str
    model_remaining: dict[str, int] = field(default_factory=dict)  # 每个模型的剩余量
    last_access_date: str = field(default_factory=lambda: date.today().isoformat())
    
    def get_remaining(self, model_name: str) -> int:
        """获取指定模型的剩余量"""
        return self.model_remaining.get(model_name, 0)
    
    def set_remaining(self, model_name: str, amount: int):
        """设置指定模型的剩余量"""
        self.model_remaining[model_name] = max(0, amount)
    
    def consume(self, model_name: str, amount: int = 1) -> bool:
        """消费指定模型的剩余量，返回是否成功"""
        current = self.get_remaining(model_name)
        if current >= amount:
            self.model_remaining[model_name] = current - amount
            return True
        return False
    
    def reset_daily_if_needed(self, default_amounts: dict[str, int] | None = None):
        """如果是新的一天，重置每日剩余量"""
        today = date.today().isoformat()
        if self.last_access_date != today:
            if default_amounts:
                # 重置为默认的每日剩余量
                for model, amount in default_amounts.items():
                    self.set_remaining(model, amount)
            self.last_access_date = today
            return True
        return False

@dataclass
class ApiKey:
    """升级版的数据类，用于存储单个 API Key 的状态"""
    key: str
    owner: str
    model_limits: dict[str, int]
    model_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    in_use: bool = field(default=False, repr=False)
    last_reset_date: str = field(default_factory=lambda: date.today().isoformat())

    def get_limit(self, model_name: str) -> int:
        """获取指定模型的限额，如果不存在则回退到 default"""
        return self.model_limits.get(model_name, self.model_limits.get("default", 0))

    def get_usage(self, model_name: str) -> int:
        """获取指定模型的使用量"""
        return self.model_usage[model_name]
    
    def reset_daily_usage_if_needed(self) -> bool:
        """如果是新的一天，重置使用量"""
        today = date.today().isoformat()
        if self.last_reset_date != today:
            self.model_usage.clear()
            self.last_reset_date = today
            return True
        return False

class ApiKeyManager:
    def __init__(self, key_path: str, user_config_path: str = "user_access.json"):
        """
        通过 JSON 文件初始化 Key 管理器。
        :param json_path: keys.json 文件的路径。
        :param user_config_path: 用户访问配置文件的路径。
        """
        self._keys: list[ApiKey] = []
        self._user_records: Dict[str, UserAccessRecord] = {}
        self._json_path = key_path
        self._user_config_path = user_config_path
        self._current_key_index = 0  # 循环索引，记录当前应该使用的密钥位置
        
        self._load_key_from_json(key_path)
        self._load_user_config()
        
        if not self._keys:
            raise ValueError(f"No valid key configurations found in {key_path}.")
            
        self._lock = threading.Lock()

    def _load_key_from_json(self, json_path: str):
        try:
            with open(json_path, 'r') as f:
                key_configs = json.load(f)
            
            for owner, key in key_configs.items():
                self._keys.append(ApiKey(
                    key=key,
                    owner=owner,
                    model_limits=MODEL_LIMITS
                ))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load or parse {json_path}: {e}") from e
    
    def _load_user_config(self):
        """加载用户访问配置"""
        try:
            if os.path.exists(self._user_config_path):
                with open(self._user_config_path, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                for username, data in user_data.items():
                    model_remaining = data['model_remaining']
                    self._user_records[username] = UserAccessRecord(username, model_remaining, date.today().isoformat())
        except Exception as e:
            print(f"Warning: Failed to load user config: {e}")
            
    def _save_user_config(self):
        """保存用户访问配置"""
        try:
            user_data = {}
            for username, record in self._user_records.items():
                user_data[username] = {
                    'model_remaining': record.model_remaining,
                    'last_access_date': record.last_access_date
                }
            
            with open(self._user_config_path, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save user config: {e}")

    def add_user(self, username: str):
        if username not in self._user_records:
            # 为新用户初始化默认的模型剩余量
            default_amounts = {
                "gemini-2.0-flash": 200,
                "gemini-2.5-flash": 250, 
                "gemini-2.5-pro": 200,
                "gemini-2.5-flash-lite": 1000,
                "Qwen2.5-VL-3B": 300,
                "Qwen2.5-VL-7B": 200,
                "Qwen2.5-VL-32B": 100
            }
            
            user_record = UserAccessRecord(username=username)
            for model, amount in default_amounts.items():
                user_record.set_remaining(model, amount)
            
            self._user_records[username] = user_record
        else:
            raise ValueError(f"Invalid username: {username} already exists")

    def _check_and_update_user_access(self, username: str, model_name: str) -> bool:
        """检查和更新用户访问记录（基于模型剩余量）"""
        today = date.today().isoformat()
        
        if username not in self._user_records:
            return False
        
        user_record = self._user_records[username]
        
        # 如果是新的一天，重置每日剩余量
        # if user_record.last_access_date != today:
        #     default_amounts = {
        #         "gemini-2.0-flash": 200,
        #         "gemini-2.5-flash": 250,
        #         "gemini-2.5-pro": 200,
        #         "gemini-2.5-flash-lite": 1000,
        #         "Qwen2.5-VL-3B": 300,
        #         "Qwen2.5-VL-7B": 200,
        #         "Qwen2.5-VL-32B": 100
        #     }
        #     user_record.reset_daily_if_needed(default_amounts)
        
        # 检查指定模型的剩余量
        remaining = user_record.get_remaining(model_name)
        if remaining <= 0:
            return False
        
        # 消耗一次使用量
        if not user_record.consume(model_name, 1):
            return False
        
        # 保存配置
        self._save_user_config()
        
        return True
    
    def _reset_daily_usage_for_all_keys(self):
        """为所有密钥重置每日使用量（如果需要）"""
        for key_obj in self._keys:
            key_obj.reset_daily_usage_if_needed()
    
    def _select_key(self, model_name: str) -> tuple[ApiKey | None, bool]:
        """【内部方法】按照循环顺序选择一个对指定模型可用的 Key
        返回: (ApiKey或None, 是否有可用额度的密钥存在)
        """
        # 首先检查和重置所有密钥的每日使用量
        # self._reset_daily_usage_for_all_keys()
        
        if not self._keys:
            return None, False
        
        has_available_quota = False  # 是否存在有可用额度的密钥
        
        # 从当前索引开始，循环遍历所有密钥
        total_keys = len(self._keys)
        start_index = self._current_key_index
        
        for i in range(total_keys):
            # 计算当前要检查的密钥索引（循环）
            key_index = (start_index + i) % total_keys
            key_obj = self._keys[key_index]
            
            limit = key_obj.get_limit(model_name)
            usage = key_obj.get_usage(model_name)
            
            # 检查是否有可用额度
            if limit > 0 and usage < limit:
                has_available_quota = True
                
                # 检查：未被使用、限额大于0、且使用量小于限额
                if not key_obj.in_use:
                    key_obj.in_use = True
                    # 更新下一次选择的起始索引（循环至下一个）
                    self._current_key_index = (key_index + 1) % total_keys
                    logger = set_logger("key_manage")
                    logger.info(f"Using key: {key_obj.key}.")
                    return key_obj, True
        
        return None, has_available_quota

    # 上下文管理器需要修改以传递 model_name 和 username
    @contextlib.contextmanager
    def get_key(self, model_name: str, username: str, wait_timeout: int = 30, check_interval: float = 0.5):
        """
        同步上下文管理器，用于获取和释放特定模型的 Key
        :param model_name: 模型名称
        :param username: 用户名
        :param wait_timeout: 等待超时时间（秒），默认30秒
        :param check_interval: 检查间隔时间（秒），默认0.5秒
        """
        selected_key = None
        start_time = time.time()
        
        try:
            while True:
                with self._lock:
                    # 检查用户访问权限
                    if not self._check_and_update_user_access(username, model_name):
                        # check the user whether update
                        self._load_user_config()
                        if not self._check_and_update_user_access(username, model_name):
                            raise UserAccessDeniedError(f"User '{username}' has exceeded access limits for model '{model_name}'.")
                    
                    selected_key, has_quota = self._select_key(model_name)
                    if selected_key:
                        break  # 成功获取密钥，退出循环
                    
                    # 如果没有可用额度的密钥，直接抛出异常
                    if not has_quota:
                        raise NoAvailableKeyError(
                            f"No API key with available quota for model '{model_name}'. "
                            f"All keys have reached their usage limit for this model."
                        )
                
                # 检查是否超时
                if time.time() - start_time > wait_timeout:
                    raise NoAvailableKeyError(
                        f"Timeout waiting for available API key for model '{model_name}'. "
                        f"All keys are currently in use. Waited {wait_timeout} seconds."
                    )
                
                # 等待一段时间后重试
                time.sleep(check_interval)
            
            yield selected_key
        finally:
            if selected_key:
                with self._lock:
                    selected_key.in_use = False
    
    @contextlib.asynccontextmanager
    async def get_key_async(self, model_name: str, username: str, wait_timeout: int = 30, check_interval: float = 0.1):
        """
        异步上下文管理器，用于获取和释放特定模型的 Key
        :param model_name: 模型名称
        :param username: 用户名
        :param wait_timeout: 等待超时时间（秒），默认30秒
        :param check_interval: 检查间隔时间（秒），默认0.1秒
        """
        selected_key = None
        start_time = time.time()
        
        try:
            while True:
                with self._lock:
                    # 检查用户访问权限
                    if not self._check_and_update_user_access(username, model_name):
                        raise UserAccessDeniedError(f"User '{username}' has exceeded access limits for model '{model_name}'.")
                    
                    selected_key, has_quota = self._select_key(model_name)
                    if selected_key:
                        break  # 成功获取密钥，退出循环
                    
                    # 如果没有可用额度的密钥，直接抛出异常
                    if not has_quota:
                        raise NoAvailableKeyError(
                            f"No API key with available quota for model '{model_name}'. "
                            f"All keys have reached their usage limit for this model."
                        )
                
                # 检查是否超时
                if time.time() - start_time > wait_timeout:
                    raise NoAvailableKeyError(
                        f"Timeout waiting for available API key for model '{model_name}'. "
                        f"All keys are currently in use. Waited {wait_timeout} seconds."
                    )
                
                # 异步等待
                await asyncio.sleep(check_interval)
            
            yield selected_key
        finally:
            if selected_key:
                with self._lock:
                    selected_key.in_use = False

    def update_usage(self, key_obj: ApiKey, model_name: str, amount: int = 1):
        """更新指定 Key 的指定模型的使用量 (默认为1次)"""
        with self._lock:
            key_obj.model_usage[model_name] += amount
    
    def get_user_status(self, username: str) -> Optional[Dict]:
        """获取用户的访问状态"""
        if username in self._user_records:
            record = self._user_records[username]
            return {
                "username": record.username,
                "model_remaining": record.model_remaining,
                "last_access_date": record.last_access_date
            }
        return None
    
    # def update_user_remaining(self, username: str, model_name: str, amount: int):
    #     """更新用户指定模型的剩余量"""
    #     if username not in self._user_records:
    #         self._user_records[username] = UserAccessRecord(username=username)
        
    #     record = self._user_records[username]
    #     record.set_remaining(model_name, amount)
    #     self._save_user_config()
    
    # def reset_user_daily_remaining(self, username: str):
    #     """重置用户的每日剩余量"""
    #     if username in self._user_records:
    #         default_amounts = {
    #             "gemini-2.0-flash": 200,
    #             "gemini-2.5-flash": 250,
    #             "gemini-2.5-pro": 100,
    #             "Qwen2.5-VL-3B": 300,
    #             "Qwen2.5-VL-7B": 200,
    #             "Qwen2.5-VL-32B": 100
    #         }
    #         # 强制重置所有模型的剩余量
    #         for model, amount in default_amounts.items():
    #             self._user_records[username].set_remaining(model, amount)
    #         self._save_user_config()
    
    def get_all_users_status(self) -> Dict[str, Dict]:
        """获取所有用户的状态"""
        result = {}
        for username, record in self._user_records.items():
            result[username] = {
                "model_remaining": record.model_remaining,
                "last_access_date": record.last_access_date
            }
        return result
    
    def get_status(self) -> Dict:
        """获取所有 Key 的当前状态，用于调试 (返回字典更通用)"""
        with self._lock:
            # 首先重置所有密钥的每日使用量
            self._reset_daily_usage_for_all_keys()
            
            status_list = []
            for k in self._keys:
                status_list.append({
                    "key": f"...{k.key[-4:]}",
                    "owner": k.owner,
                    "in_use": k.in_use,
                    "model_usage": dict(k.model_usage), # 转换 defaultdict 为 dict
                    "model_limits": k.model_limits,
                    "last_reset_date": k.last_reset_date
                })
            
            return {
                "keys": status_list,
                "users": self.get_all_users_status()
            }
