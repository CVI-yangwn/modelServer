#!/usr/bin/env python3
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import pynvml  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "错误: 缺少强制依赖 pynvml。请先执行 `pip install -r modelServer/requirements.txt`。"
    ) from exc

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.ms_manager import MsManager


class StatusMessage(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class MsTuiApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #left {
        width: 60%;
        border: solid gray;
        padding: 1;
    }
    #instance_table {
        height: 42%;
    }
    #system_panel {
        height: 58%;
        border-top: solid gray;
        padding-top: 1;
    }
    #gpu_proc_table {
        height: 1fr;
    }
    #right {
        width: 40%;
    }
    #deploy_form {
        border: solid gray;
        height: 50%;
        padding: 1;
    }
    #log_panel {
        border: solid gray;
        height: 50%;
        padding: 1;
    }
    #status_bar {
        height: 1;
        background: $boost;
        color: $text;
        padding-left: 1;
    }
    .field {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "退出"),
        Binding("n", "deploy", "部署"),
        Binding("s", "stop_selected", "停止"),
        Binding("r", "restart_selected", "重启"),
        Binding("d", "delete_selected", "删除"),
        Binding("l", "toggle_log_follow", "日志跟随"),
    ]

    selected_instance = reactive("")
    log_follow = reactive(True)

    def __init__(self):
        super().__init__()
        self.manager = MsManager(PROJECT_DIR)
        self._row_key_to_name = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Label("实例状态")
                yield DataTable(id="instance_table")
                with Container(id="system_panel"):
                    yield Label("系统监控（Lite nvitop）")
                    yield Static("加载中...", id="system_stats")
                    yield Label("GPU 进程")
                    yield DataTable(id="gpu_proc_table")
            with Vertical(id="right"):
                with Container(id="deploy_form"):
                    yield Label("部署配置")
                    model_options = [(m, m) for m in self.manager.get_registered_models()]
                    env_options = [(e, e) for e in self.manager.get_conda_envs()]
                    default_model = self.manager.default_model_name
                    default_env = env_options[0][1] if env_options else Select.BLANK
                    yield Select(model_options, value=default_model, id="model", classes="field")
                    yield Input(placeholder="实例名", id="instance_name", classes="field")
                    yield Input(value=self.manager.default_port, placeholder="端口", id="port", classes="field")
                    yield Input(value=self.manager.default_gpu_id, placeholder="GPU ID", id="gpu_id", classes="field")
                    yield Select(env_options, value=default_env, id="conda_env", classes="field")
                    yield Input(placeholder="权重路径（可空）", id="weight_path", classes="field")
                    yield Button("部署 (n)", id="deploy_btn", variant="primary")
                with Container(id="log_panel"):
                    yield Label("日志预览（选中实例）")
                    yield Static("暂无日志", id="log_view")
        yield Static("就绪", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#instance_table", DataTable)
        table.add_columns("实例名", "状态", "PID", "模型", "端口", "GPU", "Conda")
        gpu_table = self.query_one("#gpu_proc_table", DataTable)
        gpu_table.add_columns("GPU", "PID", "USER", "SM", "CPU", "TIME", "COMMAND")
        self.refresh_instances()
        self.set_interval(2, self.refresh_instances)

    def post_status(self, text: str) -> None:
        self.query_one("#status_bar", Static).update(text)

    def refresh_instances(self) -> None:
        table = self.query_one("#instance_table", DataTable)
        instances = self.manager.list_instances()
        table.clear()
        self._row_key_to_name.clear()
        for idx, item in enumerate(instances):
            pid = str(item.pid) if item.pid else "N/A"
            row_key = f"row-{idx}"
            table.add_row(item.name, item.state, pid, item.model, item.port, item.gpu_id, item.conda_env, key=row_key)
            self._row_key_to_name[row_key] = item.name

        if self.selected_instance:
            log_text = self.manager.read_log_tail(self.selected_instance, 60) if self.log_follow else "日志跟随已关闭"
            self.query_one("#log_view", Static).update(log_text)
        self.query_one("#system_stats", Static).update(self._build_system_stats_text())
        self._refresh_gpu_process_table()

    def _build_system_stats_text(self) -> str:
        lines = []
        cpu_count = os.cpu_count() or 0
        cpu_pct = self._cpu_percent()
        lines.append(f"CPU 核心数: {cpu_count}")
        lines.append(f"系统负载: {self._loadavg_text()}")
        lines.append(self._metric_line("CPU 使用率", cpu_pct))
        mem_text, mem_pct = self._memory_text()
        lines.append(self._metric_line("内存占用", mem_pct, suffix=mem_text))
        gpu_lines = self._gpu_stats_lines_with_bar()
        lines.append("")
        lines.append("GPU 状态:")
        lines.extend(gpu_lines)
        return "\n".join(lines)

    def _metric_line(self, name: str, pct: float, suffix: str = "") -> str:
        pct_color = self._pct_color(pct)
        label = f"[#8EE6B0]{name}[/#8EE6B0]"
        bar = self._progress_bar(pct, width=24)
        suffix_text = f" | {suffix}" if suffix else ""
        return f"{label}: {bar} [{pct_color}]{pct:.1f}%[/{pct_color}]{suffix_text}"

    def _progress_bar(self, pct: float, width: int = 20, fill_color: str = "#86E6A7", empty_color: str = "#2F5E3E") -> str:
        clamped = max(0.0, min(100.0, pct))
        filled = int((clamped / 100.0) * width)
        # 进度与空槽都使用实心块，仅通过颜色区分。
        return f"[{fill_color}]{'█' * filled}[/{fill_color}][{empty_color}]{'█' * (width - filled)}[/{empty_color}]"

    def _pct_color(self, pct: float) -> str:
        if pct < 50:
            return "#8EE6B0"
        if pct < 80:
            return "#FFD166"
        return "#FF6B6B"

    def _cpu_percent(self) -> float:
        try:
            result = subprocess.run(
                ["bash", "-lc", "LANG=C top -bn1 | rg '^%Cpu\\(s\\)'"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.2,
            )
            line = result.stdout.strip()
            # 例如: %Cpu(s):  2.2 us,  0.4 sy, ...
            cpu_user = float(line.split(":")[1].split("us")[0].strip())
            return max(cpu_user, 0.0)
        except Exception:
            return 0.0

    def _loadavg_text(self) -> str:
        try:
            one, five, fifteen = os.getloadavg()
            return f"{one:.2f} / {five:.2f} / {fifteen:.2f} (1m/5m/15m)"
        except Exception:
            return "N/A"

    def _memory_text(self) -> tuple[str, float]:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                info = {}
                for line in f:
                    key, value = line.split(":", 1)
                    info[key.strip()] = value.strip()
            total_kb = int(info["MemTotal"].split()[0])
            avail_kb = int(info["MemAvailable"].split()[0])
            used_kb = max(total_kb - avail_kb, 0)
            pct = (used_kb / total_kb * 100) if total_kb else 0
            return f"{used_kb // 1024} MiB / {total_kb // 1024} MiB", pct
        except Exception:
            return "N/A", 0.0

    def _gpu_stats_lines_with_bar(self) -> list[str]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1.2)
            rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
            if not rows:
                return ["未检测到 GPU 设备"]
            output = []
            for row in rows:
                parts = [p.strip() for p in row.split(",")]
                if len(parts) < 5:
                    output.append(row)
                    continue
                idx, name, util, mem_used, mem_total = parts[:5]
                util_f = float(util)
                mem_total_f = float(mem_total) if float(mem_total) > 0 else 1.0
                mem_pct = float(mem_used) / mem_total_f * 100
                util_bar = self._progress_bar(util_f, width=16)
                mem_bar = self._progress_bar(mem_pct, width=16)
                util_color = self._pct_color(util_f)
                mem_color = self._pct_color(mem_pct)
                output.append(
                    f"[#8EE6B0]GPU{idx} {name}[/#8EE6B0] | "
                    f"[#8EE6B0]SM[/#8EE6B0] {util_bar} [{util_color}]{util_f:.0f}%[/{util_color}] | "
                    f"[#8EE6B0]MEM[/#8EE6B0] {mem_bar} [{mem_color}]{mem_pct:.0f}%[/{mem_color}] "
                    f"({mem_used}/{mem_total} MiB)"
                )
            return output
        except Exception:
            return ["未检测到 nvidia-smi（或 GPU 不可用）"]

    def _refresh_gpu_process_table(self) -> None:
        table = self.query_one("#gpu_proc_table", DataTable)
        table.clear()
        rows = self._gpu_process_rows()
        if not rows:
            table.add_row("-", "-", "-", "-", "-", "-", "无 GPU 进程")
            return
        for row in rows:
            table.add_row(*row)

    def _gpu_process_rows(self) -> list[tuple[str, str, str, str, str, str, str]]:
        # 通过 query-compute-apps 获取稳定的 pid 和 gpu_id，再结合 pmon 补 SM。
        try:
            gpu_map_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            app_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
        except Exception:
            return []

        uuid_to_idx: dict[str, str] = {}
        for raw in gpu_map_result.stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[1]] = parts[0]

        sm_map = self._gpu_sm_by_pid()

        rows = []
        for raw in app_result.stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            gpu_uuid, pid, process_name = parts[:3]
            if pid in {"-", "0", ""}:
                continue
            gpu_id = uuid_to_idx.get(gpu_uuid, "?")
            sm = sm_map.get((gpu_id, pid), "-")
            user, cpu, etime, command = self._ps_info(pid, fallback_cmd=process_name)
            sm_cell = self._colorize_percent(sm)
            cpu_cell = self._colorize_percent(cpu.replace("%", ""))
            rows.append((gpu_id, pid, user, sm_cell, cpu_cell, etime, command))

        rows.sort(key=self._gpu_proc_sort_key, reverse=True)
        return rows

    def _gpu_proc_sort_key(self, row: tuple[str, str, str, str, str, str, str]) -> tuple[float, float]:
        sm_num = self._parse_pct_from_cell(row[3])
        cpu_num = self._parse_pct_from_cell(row[4])
        return (sm_num, cpu_num)

    def _parse_pct_from_cell(self, value: str) -> float:
        clean = (
            value.replace("[red]", "")
            .replace("[/red]", "")
            .replace("[yellow]", "")
            .replace("[/yellow]", "")
            .replace("[green]", "")
            .replace("[/green]", "")
            .replace("%", "")
            .strip()
        )
        try:
            return float(clean)
        except Exception:
            return -1.0

    def _colorize_percent(self, raw: str) -> str:
        try:
            pct = float(raw)
        except Exception:
            return "-"
        color = self._pct_color(pct)
        return f"[{color}]{pct:.1f}%[/{color}]"

    def _gpu_sm_by_pid(self) -> dict[tuple[str, str], str]:
        try:
            pmon = subprocess.run(
                ["nvidia-smi", "pmon", "-c", "1"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
        except Exception:
            return {}

        sm_map: dict[tuple[str, str], str] = {}
        for raw in pmon.stdout.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 4:
                continue
            gpu_id, pid, sm = cols[0], cols[1], cols[3]
            if pid in {"-", "0"}:
                continue
            sm_map[(gpu_id, pid)] = sm
        return sm_map

    def _ps_info(self, pid: str, fallback_cmd: str) -> tuple[str, str, str, str]:
        user = self._ps_single(pid, "user") or self._user_from_proc(pid) or "-"
        cpu = self._ps_single(pid, "pcpu")
        etime = self._ps_single(pid, "etime") or self._elapsed_from_proc(pid) or "-"
        command = self._ps_single(pid, "args") or self._command_from_proc(pid) or fallback_cmd
        if cpu:
            try:
                cpu = f"{float(cpu):.1f}%"
            except Exception:
                cpu = "-"
        else:
            cpu = "-"
        command = command[:62] + "..." if len(command) > 65 else command
        return user, cpu, etime, command

    def _ps_single(self, pid: str, field: str) -> str:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", f"{field}="],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode != 0:
                return ""
            return result.stdout.strip()
        except Exception:
            return ""

    def _user_from_proc(self, pid: str) -> str:
        try:
            status_path = Path("/proc") / str(pid) / "status"
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])
                    return pwd.getpwuid(uid).pw_name
        except Exception:
            return ""
        return ""

    def _command_from_proc(self, pid: str) -> str:
        try:
            cmdline_path = Path("/proc") / str(pid) / "cmdline"
            raw = cmdline_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", " ").strip()
            return raw
        except Exception:
            return ""

    def _elapsed_from_proc(self, pid: str) -> str:
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8", errors="ignore")
            parts = stat.split()
            start_ticks = int(parts[21])
            hertz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
            elapsed = max(0, int(uptime - (start_ticks / hertz)))
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            return f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            return ""

    @on(DataTable.RowSelected, "#instance_table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key)
        self.selected_instance = self._row_key_to_name.get(row_key, "")
        if self.selected_instance:
            self.query_one("#instance_name", Input).value = self.selected_instance
            self.query_one("#log_view", Static).update(self.manager.read_log_tail(self.selected_instance, 60))

    @on(Button.Pressed, "#deploy_btn")
    def on_deploy_pressed(self) -> None:
        self.action_deploy()

    def _get_form_values(self):
        model = str(self.query_one("#model", Select).value or "").strip()
        instance_name = self.query_one("#instance_name", Input).value.strip()
        port = self.query_one("#port", Input).value.strip() or self.manager.default_port
        gpu_id = self.query_one("#gpu_id", Input).value.strip() or self.manager.default_gpu_id
        conda_env = str(self.query_one("#conda_env", Select).value or "").strip()
        weight_path = self.query_one("#weight_path", Input).value.strip()
        return model, instance_name, port, gpu_id, conda_env, weight_path

    def action_deploy(self) -> None:
        model, instance_name, port, gpu_id, conda_env, weight_path = self._get_form_values()
        if not instance_name:
            self.post_status("部署失败：实例名不能为空")
            return
        if not conda_env:
            self.post_status("部署失败：Conda 环境不能为空")
            return
        ok, msg = self.manager.start_instance(
            instance_name=instance_name,
            model_name=model,
            port=port,
            gpu_id=gpu_id,
            conda_env_name=conda_env,
            weight_path=weight_path,
            wait_for_ready=False,
        )
        self.selected_instance = instance_name
        self.post_status(msg)
        self.refresh_instances()

    def _require_selection(self) -> Optional[str]:
        if not self.selected_instance:
            self.post_status("请先在左侧选择一个实例")
            return None
        return self.selected_instance

    def action_stop_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        _, msg = self.manager.stop_instance(name)
        self.post_status(msg)
        self.refresh_instances()

    def action_restart_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        _, msg = self.manager.restart_instance(name, wait_for_ready=False)
        self.post_status(msg)
        self.refresh_instances()

    def action_delete_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        _, msg = self.manager.delete_instance(name)
        if self.selected_instance == name:
            self.selected_instance = ""
        self.post_status(msg)
        self.refresh_instances()

    def action_toggle_log_follow(self) -> None:
        self.log_follow = not self.log_follow
        state = "开启" if self.log_follow else "关闭"
        self.post_status(f"日志跟随已{state}")
        self.refresh_instances()


def main() -> int:
    # 强制依赖检查：确保 pynvml 在运行期可用。
    pynvml.nvmlInit()
    pynvml.nvmlShutdown()
    app = MsTuiApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
