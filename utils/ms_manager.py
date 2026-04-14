import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import name_to_model_class


READY_FLAG = "Test Successfully!"


@dataclass
class InstanceStatus:
    name: str
    model: str
    port: str
    gpu_id: str
    conda_env: str
    weight_path: str
    pid: Optional[int]
    state: str
    log_file: str
    instance_dir: str


class MsManager:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir).resolve()
        self.instance_dir = self.project_dir / "logs"
        self.script_name = "modelServer.py"
        self.default_model_name = "Qwen2.5-VL-7B"
        self.default_port = "9960"
        self.default_gpu_id = "0"
        self.start_timeout = 1800
        self.instance_dir.mkdir(parents=True, exist_ok=True)

    def get_registered_models(self) -> List[str]:
        return sorted(name_to_model_class.keys())

    def get_conda_envs(self) -> List[str]:
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return []

        envs = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            envs.append(line.split()[0])
        return envs

    def _find_activate_script(self) -> Optional[str]:
        conda_bin = shutil.which("conda")
        if conda_bin:
            try:
                base_dir = (
                    subprocess.run(
                        ["conda", "info", "--base"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    .stdout.strip()
                )
                candidate = Path(base_dir) / "bin" / "activate"
                if candidate.exists():
                    return str(candidate)
            except Exception:
                pass

        for candidate in (
            Path.home() / "miniconda3" / "bin" / "activate",
            Path.home() / "anaconda3" / "bin" / "activate",
        ):
            if candidate.exists():
                return str(candidate)
        return None

    def _instance_paths(self, instance_name: str) -> Dict[str, Path]:
        base = self.instance_dir / instance_name
        return {
            "base": base,
            "conf": base / f"{instance_name}.conf",
            "pid": base / f"{instance_name}.pid",
            "log": base / f"{instance_name}.log",
        }

    def _read_conf(self, conf_file: Path) -> Dict[str, str]:
        result: Dict[str, str] = {}
        if not conf_file.exists():
            return result
        for raw in conf_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            result[key.strip()] = value
        return result

    def _write_conf(self, conf_file: Path, conf: Dict[str, str]) -> None:
        lines = [
            f'MODEL_NAME="{conf["MODEL_NAME"]}"',
            f'PORT={conf["PORT"]}',
            f'GPU_ID={conf["GPU_ID"]}',
            f"CONDA_ENV_NAME={conf['CONDA_ENV_NAME']}",
            f"WEIGHT_PATH={conf['WEIGHT_PATH']}",
        ]
        conf_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _read_pid(self, pid_file: Path) -> Optional[int]:
        if not pid_file.exists():
            return None
        try:
            return int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def _is_pid_running(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def is_running(self, instance_name: str) -> bool:
        paths = self._instance_paths(instance_name)
        pid = self._read_pid(paths["pid"])
        return self._is_pid_running(pid)

    def _is_ready(self, log_file: Path) -> bool:
        if not log_file.exists():
            return False
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        return READY_FLAG in content

    def list_instances(self) -> List[InstanceStatus]:
        statuses: List[InstanceStatus] = []
        if not self.instance_dir.exists():
            return statuses
        for item in sorted(self.instance_dir.iterdir()):
            if not item.is_dir():
                continue
            name = item.name
            paths = self._instance_paths(name)
            conf = self._read_conf(paths["conf"])
            pid = self._read_pid(paths["pid"])
            running = self._is_pid_running(pid)
            ready = self._is_ready(paths["log"])
            if running and ready:
                state = "running"
            elif running:
                state = "starting"
            elif pid is not None and not running:
                state = "stopped"
            else:
                state = "stopped"
            statuses.append(
                InstanceStatus(
                    name=name,
                    model=conf.get("MODEL_NAME", ""),
                    port=conf.get("PORT", ""),
                    gpu_id=conf.get("GPU_ID", ""),
                    conda_env=conf.get("CONDA_ENV_NAME", ""),
                    weight_path=conf.get("WEIGHT_PATH", ""),
                    pid=pid if running else None,
                    state=state,
                    log_file=str(paths["log"]),
                    instance_dir=str(paths["base"]),
                )
            )
        return statuses

    def get_instance(self, instance_name: str) -> Optional[InstanceStatus]:
        for instance in self.list_instances():
            if instance.name == instance_name:
                return instance
        return None

    def _build_start_cmd(
        self,
        model_name: str,
        port: str,
        gpu_id: str,
        conda_env_name: str,
        weight_path: str,
    ) -> str:
        activate_script = self._find_activate_script()
        if not activate_script:
            raise RuntimeError("无法找到 Conda activate 脚本")
        cmd_parts = [
            f"source {shlex.quote(activate_script)} {shlex.quote(conda_env_name)}",
            f"python -u {shlex.quote(self.script_name)} --model {shlex.quote(model_name)} --port {shlex.quote(str(port))}",
        ]
        if weight_path:
            cmd_parts[-1] += f" --weight_path {shlex.quote(weight_path)}"
        return (
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))}; "
            "export OMP_NUM_THREADS=8; "
            + " && ".join(cmd_parts)
        )

    def start_instance(
        self,
        instance_name: str,
        model_name: Optional[str] = None,
        port: Optional[str] = None,
        gpu_id: Optional[str] = None,
        conda_env_name: Optional[str] = None,
        weight_path: Optional[str] = None,
        wait_for_ready: bool = False,
    ) -> Tuple[bool, str]:
        model_name = model_name or self.default_model_name
        port = str(port or self.default_port)
        gpu_id = str(gpu_id or self.default_gpu_id)
        weight_path = weight_path or ""

        if self.is_running(instance_name):
            return True, f"实例 {instance_name} 已在运行中"
        if not conda_env_name:
            return False, "缺少 conda 环境名称"

        paths = self._instance_paths(instance_name)
        paths["base"].mkdir(parents=True, exist_ok=True)
        paths["log"].write_text("", encoding="utf-8")
        self._write_conf(
            paths["conf"],
            {
                "MODEL_NAME": model_name,
                "PORT": port,
                "GPU_ID": gpu_id,
                "CONDA_ENV_NAME": conda_env_name,
                "WEIGHT_PATH": weight_path,
            },
        )

        full_command = self._build_start_cmd(model_name, port, gpu_id, conda_env_name, weight_path)
        with paths["log"].open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                ["bash", "-lc", full_command],
                cwd=str(self.project_dir),
                stdout=log_handle,
                stderr=log_handle,
                preexec_fn=os.setsid,
            )
        paths["pid"].write_text(str(process.pid), encoding="utf-8")

        if not wait_for_ready:
            return True, f"实例 {instance_name} 启动中 (PID: {process.pid})"

        elapsed = 0
        while elapsed < self.start_timeout:
            if not self._is_pid_running(process.pid):
                return False, f"实例 {instance_name} 在启动阶段退出，请检查日志"
            if self._is_ready(paths["log"]):
                return True, f"实例 {instance_name} 启动成功 (PID: {process.pid})"
            time.sleep(2)
            elapsed += 2
        return False, f"实例 {instance_name} 启动超时"

    def stop_instance(self, instance_name: str) -> Tuple[bool, str]:
        paths = self._instance_paths(instance_name)
        pid = self._read_pid(paths["pid"])
        if not self._is_pid_running(pid):
            if paths["pid"].exists():
                paths["pid"].unlink()
            return True, f"实例 {instance_name} 未在运行"

        assert pid is not None
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            pass

        for _ in range(10):
            if not self._is_pid_running(pid):
                break
            time.sleep(1)
        if self._is_pid_running(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except Exception:
                pass
        if paths["pid"].exists():
            paths["pid"].unlink()
        return True, f"实例 {instance_name} 已停止"

    def restart_instance(self, instance_name: str, wait_for_ready: bool = False) -> Tuple[bool, str]:
        paths = self._instance_paths(instance_name)
        conf = self._read_conf(paths["conf"])
        if not conf:
            return False, f"找不到实例 {instance_name} 配置"
        self.stop_instance(instance_name)
        return self.start_instance(
            instance_name=instance_name,
            model_name=conf.get("MODEL_NAME", self.default_model_name),
            port=conf.get("PORT", self.default_port),
            gpu_id=conf.get("GPU_ID", self.default_gpu_id),
            conda_env_name=conf.get("CONDA_ENV_NAME", ""),
            weight_path=conf.get("WEIGHT_PATH", ""),
            wait_for_ready=wait_for_ready,
        )

    def delete_instance(self, instance_name: str) -> Tuple[bool, str]:
        paths = self._instance_paths(instance_name)
        if not paths["base"].exists():
            return True, f"实例 {instance_name} 不存在"
        self.stop_instance(instance_name)
        shutil.rmtree(paths["base"], ignore_errors=True)
        return True, f"实例 {instance_name} 已删除"

    def read_log_tail(self, instance_name: str, lines: int = 80) -> str:
        paths = self._instance_paths(instance_name)
        if not paths["log"].exists():
            return "暂无日志"
        all_lines = paths["log"].read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(all_lines[-lines:]) if all_lines else "暂无日志"
