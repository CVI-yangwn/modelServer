# server.py
from turtle import Vec2D
import tornado
import tornado.ioloop
import tornado.web
import asyncio
import json
import time
import uuid
import os
from concurrent.futures import ThreadPoolExecutor

from logger import set_logger, logger_output
from key_manage import ApiKeyManager, NoAvailableKeyError, UserAccessDeniedError

from utils import *

def post_qwen(data):
    # 实现你的 Qwen 调用逻辑
    return {"model": "qwen", "response": "This is a dummy response from Qwen."}

def get_api_key_from_request(request_handler: tornado.web.RequestHandler) -> str | None:
    auth_header = request_handler.request.headers.get("Authorization")
    
    if not auth_header:
        return None
    parts = auth_header.split()
    
    api_key = parts[1]
    return api_key

class BaseHandler(tornado.web.RequestHandler):

    def initialize(self, executor: ThreadPoolExecutor, key_manager: ApiKeyManager | None = None):
        self.executor = executor
        self.key_manager = key_manager

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()

    async def post(self):
        # rename api key to username
        username = get_api_key_from_request(self)
        if not (username and self.key_manager.get_user_status(username)):
            self.set_status(401)
            self.write({"error": "Missing or Wrong Authorization header with API key."})
            return

        try:
            request_data = json.loads(self.request.body)
        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"error": "Invalid JSON format."})
            return

        model_name = request_data.get("model")
        messages = request_data.get("messages")
        stream_mode = request_data.get("stream", False)

        if not all([model_name, messages]):
            self.set_status(400)
            self.write({"error": "Missing 'model' or 'messages' in request body."})
            return

        if model_name not in SUPPORT_MODELS:
            self.set_status(400)
            self.write({"error": f"Model '{model_name}' is not supported. Available models: {SUPPORT_MODELS}"})
            return
        
        # 检查用户访问权限 - 这里先进行基本检查，实际密钥获取在process方法中
        try:
            # 预检查用户状态
            user_status = self.key_manager.get_user_status(username)
            if user_status:
                # 简单检查是否已经超过总限制
                if user_status['model_remaining'][model_name] <= 0:
                    self.set_status(403)
                    self.write({"error": f"User '{username}' has exceeded total usage limit. Contact administrator."})
                    return
        except Exception as e:
            print(f"Error checking user status: {e}")
        
        # 存储用户名到请求对象中，供后续使用
        self.username = username

        if stream_mode:
            await self.handle_stream_request(request_data)
        else:
            await self.handle_non_stream_request(request_data)

    @property
    def name(self):
        return self.__class__.__name__.split("Handler")[0]

    async def handle_non_stream_request(self, request_data: dict):
        model_name = request_data.get("model")
        messages = request_data.get("messages")

        async def run_in_executor_and_get_result():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.process_non_stream,
                request_data
            )

        try:
            # 2. 现在把这个新的、真正的协程传递给 create_task
            task = asyncio.create_task(run_in_executor_and_get_result())

            # --- 心跳保活循环 (这段逻辑保持不变) ---
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5)
                except asyncio.TimeoutError:
                    # 5 秒过去了，任务还没完成，发送心跳
                    # 使用空格作为心跳对大多数JSON解析器是安全的
                    self.write(" ")
                    await self.flush()
            
            # --- 任务完成 ---
            if task.exception():
                raise task.exception()
                
            response_text = task.result()
            
            if not self._headers_written:
                 self.set_header("Content-Type", "application/json; charset=UTF-8")

            self.write(self.create_complete_response(response_text, model_name))
            
        except Exception as e:
            print(f"Error in non-stream request: {e}") # 打印详细错误信息
            if not self._finished:
                self.set_status(503)
                self.write({"error": f"Handle API request failed: {str(e)}"})
        
        finally:
            if not self._finished:
                self.finish()

    async def handle_stream_request(self, request_data: dict):
        model_name = request_data.get("model")
        
        self.set_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.set_header('Cache-Control', 'no-cache')
        
        chat_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        sse_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": ""
                }]
            }

        try:
            self.write(f"data: {json.dumps(sse_chunk, ensure_ascii=False)}\n\n")
            await self.flush()
            async for chunk in self.process_stream(request_data):
                sse_chunk = self.format_chunk_to_sse(chunk, chat_id, created_time, model_name)
                self.write(f"data: {json.dumps(sse_chunk, ensure_ascii=False)}\n\n")
                await self.flush()
            # 发送流结束标志
            self.write("data: [DONE]\n\n")
            await self.flush()

        except Exception as e:
            if not self._headers_written:
                self.set_status(500)
                self.set_header('Content-Type', 'application/json')
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                self.write(json.dumps(error_response))
            else:
                # 如果头已经发送，我们无能为力，只能在日志中记录
                # 连接会被非正常关闭
                pass
        
        finally:
            if not self._finished:
                self.finish()

    def create_complete_response(self, text: str, model_name: str):
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
        }

    def format_chunk_to_sse(self, chunk, chat_id, created_time, model_name):
        delta = chunk.choices[0].delta
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": delta.content or ""},
                "finish_reason": chunk.choices[0].finish_reason
            }]
        }

    def process_non_stream(self, model_name: str, messages: list):
        raise NotImplementedError("Non stream processing not implemented")

    async def process_stream(self, model_name: str, messages: list):
        raise NotImplementedError("Stream processing not implemented")


class localHandler(BaseHandler):

    def process_non_stream(self, request_data: dict):
        """非流式处理，集成密钥管理"""
        username = getattr(self, 'username', 'unknown_user')
        model_name = request_data.get("model")
        messages = request_data.get("messages")
        try:
            if self.key_manager._check_and_update_user_access(username, model_name):
                
                result = post_local_sync(request_data)
                
                return result
                
        except (NoAvailableKeyError, UserAccessDeniedError) as e:
            # self.set_status(503)
            # self.write(chunk={"error": f"Key management error for user {username}: {e}"})
            raise e
        except Exception as e:
            # self.set_status(503)
            # self.write({"error": f"Model call error for user {username}: {e}"})
            raise ValueError(f"Local inference error -- {e}")


class webHandler(BaseHandler):

    def process_non_stream(self, request_data: dict):
        """非流式处理，集成密钥管理"""
        username = getattr(self, 'username', 'unknown_user')
        model_name = request_data.get("model")
        messages = request_data.get("messages")
        try:
            # 使用密钥管理器获取密钥
            with self.key_manager.get_key(model_name, username, wait_timeout=30) as api_key:
                
                result = post_gemini_sync(request_data, api_key.key)
                
                self.key_manager.update_usage(api_key, model_name, 1)
                print(f"Updated usage for user {username}, key ...{api_key.key[-4:]}, model {model_name}")
                
                return result
                
        except (NoAvailableKeyError, UserAccessDeniedError) as e:
            # self.set_status(503)
            # self.write(chunk={"error": f"Key management error for user {username}: {e}"})
            raise e
        except Exception as e:
            # self.set_status(503)
            # self.write({"error": f"Model call error for user {username}: {e}"})
            raise ValueError(f"Official Gemini error -- {e}")

    async def process_stream(self, request_data: dict):
        """流式处理，集成密钥管理"""
        username = getattr(self, 'username', 'unknown_user')
        model_name = request_data.get("model")
        messages = request_data.get("messages")

        try:
            async with self.key_manager.get_key_async(model_name, username, wait_timeout=30) as api_key:
                print(f"User {username} using key ...{api_key.key[-4:]} for streaming model {model_name}")
                
                try:
                    async for chunk in post_gemini_stream(request_data, api_key.key):
                        yield chunk
                except Exception as model_error:
                    print(f"Model stream call error for user {username}: {model_error}")
                    # 不要 raise，而是 yield 一个错误对象
                    yield {"error": str(model_error), "error_type": "model_error"}
                    # yield 之后可以 return，以确保生成器停止
                    return
                
                self.key_manager.update_usage(api_key, model_name, 1)
                print(f"Updated usage for user {username}, key ...{api_key.key[-4:]}, model {model_name}")
                
        except (NoAvailableKeyError, UserAccessDeniedError) as e:
            print(f"Key management error for user {username}: {e}")
            # raise Exception(f"Service unavailable: {e}")
            yield {"error": f"Service unavailable: {e}"}
        except Exception as e:
            print(f"Model stream call error for user {username}: {e}")
            yield {"error": f"Model stream call error: {e}"}
            # raise


class TransServer:
    def __init__(self, port, server_address):
        self.port = port
        self.address = server_address
        self.key_manager = ApiKeyManager(os.path.join(_current_dir, "config/keys.json"),
                                                        os.path.join(_current_dir, "config/user_access.json"))

    def build(self):
        executor = ThreadPoolExecutor(max_workers=10)
        app = tornado.web.Application([
            (r"/local/chat/completions", localHandler, dict(executor=executor, key_manager=self.key_manager)),
            (r"/v1/chat/completions", webHandler, dict(executor=executor, key_manager=self.key_manager))
        ])
        app.listen(self.port, address=self.address)
        logger = set_logger(name="trans_server")
        logger.info(f"Server starting on http://{self.address or '127.0.0.1'}:{self.port}")
        logger.info("API endpoint available at /v1/chat/completions")
        logger.info(f"Key manager initialized with {len(self.key_manager._keys)} API keys")


async def main():
    server = TransServer(port=2559, server_address="0.0.0.0")
    server.build()
    await asyncio.Event().wait()


if __name__ == "__main__":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        print("Starting Tornado server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        tornado.ioloop.IOLoop.current().stop()