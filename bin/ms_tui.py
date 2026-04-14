#!/usr/bin/env python3
import os
import pwd
import subprocess
import sys
import time
import threading
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
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static
from rich.markup import escape
from rich.text import Text

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.ms_manager import MsManager


class StatusMessage(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class ConfirmDialog(ModalScreen[bool]):
    def __init__(self, title: str, message: str):
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm_dialog"):
            yield Label(self._title)
            yield Static(self._message)
            with Horizontal():
                yield Button("取消", id="cancel", variant="default")
                yield Button("确定", id="confirm", variant="error")

    def on_mount(self) -> None:
        self.query_one("#cancel", Button).focus()

    def _focused_button_id(self) -> str:
        focused = self.focused
        if isinstance(focused, Button):
            return focused.id or "cancel"
        return "cancel"

    def action_focus_left(self) -> None:
        current = self._focused_button_id()
        target = "#cancel" if current == "confirm" else "#confirm"
        self.query_one(target, Button).focus()

    def action_focus_right(self) -> None:
        current = self._focused_button_id()
        target = "#confirm" if current == "cancel" else "#cancel"
        self.query_one(target, Button).focus()

    def action_submit(self) -> None:
        current = self._focused_button_id()
        self.dismiss(current == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_key(self, event) -> None:
        key = event.key
        if key == "left":
            self.action_focus_left()
            event.stop()
        elif key == "right":
            self.action_focus_right()
            event.stop()
        elif key == "enter":
            self.action_submit()
            event.stop()
        elif key == "escape":
            self.action_cancel()
            event.stop()

    @on(Button.Pressed, "#confirm")
    def _confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(False)


class CellEditDialog(ModalScreen[Optional[str]]):
    def __init__(self, title: str, initial_value: str):
        super().__init__()
        self._title = title
        self._initial_value = initial_value

    def compose(self) -> ComposeResult:
        with Container(id="confirm_dialog"):
            yield Label(self._title)
            yield Input(value=self._initial_value, id="cell_input")
            with Horizontal():
                yield Button("取消", id="cancel", variant="default")
                yield Button("保存", id="save", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#cell_input", Input).focus()

    @on(Button.Pressed, "#save")
    def _save(self) -> None:
        value = self.query_one("#cell_input", Input).value
        self.dismiss(value)

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == "enter":
            self._save()
            event.stop()
        elif event.key == "escape":
            self._cancel()
            event.stop()


class ChoiceEditDialog(ModalScreen[Optional[str]]):
    BINDINGS = [
        Binding("enter", "submit_choice", "保存"),
        Binding("escape", "cancel_choice", "取消"),
    ]

    def __init__(self, title: str, options: list[str], initial_value: str):
        super().__init__()
        self._title = title
        self._options = options
        self._initial_value = initial_value
        self._auto_commit_enabled = False

    def compose(self) -> ComposeResult:
        with Container(id="confirm_dialog"):
            yield Label(self._title)
            select_options = [(item, item) for item in self._options]
            default = self._initial_value if self._initial_value in self._options else (self._options[0] if self._options else Select.BLANK)
            yield Select(select_options, value=default, id="choice_select")
            with Horizontal():
                yield Button("取消", id="cancel", variant="default")
                yield Button("保存", id="save", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#choice_select", Select).focus()
        # 避免初始渲染阶段误触发 changed 事件导致弹窗立即关闭。
        self.set_timer(0.05, self._enable_auto_commit)

    def _enable_auto_commit(self) -> None:
        self._auto_commit_enabled = True

    @on(Button.Pressed, "#save")
    def _save(self) -> None:
        value = self.query_one("#choice_select", Select).value
        self.dismiss("" if value is Select.BLANK else str(value))

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Select.Changed, "#choice_select")
    def _auto_save_on_select(self, event: Select.Changed) -> None:
        # 在下拉中按 Enter 选中后会触发 changed，直接保存关闭。
        if not self._auto_commit_enabled:
            return
        value = event.value
        self.dismiss("" if value is Select.BLANK else str(value))

    def action_submit_choice(self) -> None:
        self._save()

    def action_cancel_choice(self) -> None:
        self._cancel()


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
    #log_panel {
        border: solid gray;
        height: 100%;
        padding: 1;
    }
    #status_bar {
        height: 1;
        background: $boost;
        color: $text;
        padding-left: 1;
    }
    #confirm_dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    .field {
        margin-bottom: 1;
    }
    .row {
        height: auto;
        margin-bottom: 1;
    }
    .half {
        width: 1fr;
        margin-right: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "退出"),
        Binding("n", "deploy", "部署"),
        Binding("e", "edit_cell", "编辑单元格"),
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
        self._row_index_to_name = {}
        self._monitor_lock = threading.Lock()
        self._monitor_paused = False
        self._monitor_stop = False
        self._cached_system_stats = "加载中..."
        self._cached_gpu_rows: list[tuple[str, str, str, str, str, str, str]] = []
        self._edit_lock_until = 0.0
        self._instances_cache = []
        self._edited_rows: dict[str, dict[str, str]] = {}
        self._draft_row = {
            "name": "",
            "model": self.manager.default_model_name,
            "port": str(self.manager.default_port),
            "gpu_id": str(self.manager.default_gpu_id),
            "conda_env": "",
            "weight_path": "",
        }

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
                with Container(id="log_panel"):
                    yield Label("日志预览（选中实例） | e:编辑单元格 n:部署空白行")
                    yield Static("暂无日志", id="log_view")
        yield Static("就绪", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#instance_table", DataTable)
        table.add_columns("", "实例名", "模型", "端口", "GPU", "Conda", "路径", "状态", "PID")
        gpu_table = self.query_one("#gpu_proc_table", DataTable)
        gpu_table.add_columns("GPU", "PID", "USER", "SM", "CPU", "TIME", "COMMAND")
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        self.refresh_instances(force_table_refresh=True)
        self.set_interval(1, self._refresh_runtime_panels_tick)

    def on_unmount(self) -> None:
        self._monitor_stop = True

    def post_status(self, text: str) -> None:
        self.query_one("#status_bar", Static).update(text)

    def _set_log_view(self, log_text: str) -> None:
        # 强制按纯文本渲染，彻底避免 Rich markup 解析报错。
        self.query_one("#log_view", Static).update(Text(log_text))

    def _refresh_runtime_panels_tick(self) -> None:
        self._refresh_side_panels_only()

    def refresh_instances(self, force_table_refresh: bool = False) -> None:
        if not force_table_refresh and self._is_editing_form():
            self._refresh_side_panels_only()
            return

        table = self.query_one("#instance_table", DataTable)
        cursor_name_before = None
        cursor_row = getattr(table, "cursor_row", None)
        if isinstance(cursor_row, int):
            cursor_name_before = self._row_index_to_name.get(cursor_row)

        instances = self.manager.list_instances()
        self._instances_cache = instances
        table.clear()
        self._row_key_to_name.clear()
        self._row_index_to_name.clear()
        cursor_row_after = None
        for idx, item in enumerate(instances):
            row_conf = self._get_row_config(item.name)
            pid = str(item.pid) if item.pid else "N/A"
            row_key = f"row-{idx}"
            selected_mark = "☑" if item.name == self.selected_instance else "☐"
            table.add_row(
                selected_mark,
                escape(row_conf["name"]),
                escape(row_conf["model"]),
                escape(row_conf["port"]),
                escape(row_conf["gpu_id"]),
                escape(row_conf["conda_env"]),
                escape(row_conf["weight_path"]),
                escape(item.state),
                escape(pid),
                key=row_key,
            )
            self._row_key_to_name[row_key] = item.name
            self._row_index_to_name[idx] = item.name
            if cursor_name_before and item.name == cursor_name_before:
                cursor_row_after = idx

        # 固定追加空白实例行，供直接编辑后部署。
        new_row_idx = len(instances)
        table.add_row(
            "□",
            escape(self._draft_row["name"]),
            escape(self._draft_row["model"]),
            escape(self._draft_row["port"]),
            escape(self._draft_row["gpu_id"]),
            escape(self._draft_row["conda_env"]),
            escape(self._draft_row["weight_path"]),
            "new",
            "-",
            key="row-new",
        )
        self._row_index_to_name[new_row_idx] = "__new__"

        if self.selected_instance:
            log_text = self.manager.read_log_tail(self.selected_instance, 60) if self.log_follow else "日志跟随已关闭"
            self._set_log_view(log_text)
        with self._monitor_lock:
            system_stats = self._cached_system_stats
            gpu_rows = list(self._cached_gpu_rows)
        self.query_one("#system_stats", Static).update(system_stats)
        self._render_gpu_process_table(gpu_rows)

    def _refresh_side_panels_only(self) -> None:
        if self.selected_instance:
            log_text = self.manager.read_log_tail(self.selected_instance, 60) if self.log_follow else "日志跟随已关闭"
            self._set_log_view(log_text)
        with self._monitor_lock:
            system_stats = self._cached_system_stats
            gpu_rows = list(self._cached_gpu_rows)
        self.query_one("#system_stats", Static).update(system_stats)
        self._render_gpu_process_table(gpu_rows)

    def _get_row_config(self, instance_name: str) -> dict[str, str]:
        instance = next((i for i in self._instances_cache if i.name == instance_name), None)
        base = {
            "name": instance_name,
            "model": instance.model if instance else "",
            "port": str(instance.port) if instance else "",
            "gpu_id": str(instance.gpu_id) if instance else "",
            "conda_env": instance.conda_env if instance else "",
            "weight_path": instance.weight_path if instance else "",
        }
        override = self._edited_rows.get(instance_name, {})
        for k, v in override.items():
            base[k] = v
        return base

    def _is_editing_form(self) -> bool:
        now = time.monotonic()
        if now < self._edit_lock_until:
            return True
        return self.screen.is_modal

    def _touch_edit_lock(self, seconds: float = 3.0) -> None:
        self._edit_lock_until = max(self._edit_lock_until, time.monotonic() + seconds)

    def _monitor_loop(self) -> None:
        while not self._monitor_stop:
            if self._monitor_paused:
                time.sleep(0.2)
                continue
            try:
                stats = self._build_system_stats_text()
                rows = self._gpu_process_rows()
                with self._monitor_lock:
                    self._cached_system_stats = stats
                    self._cached_gpu_rows = rows
            except Exception:
                pass
            time.sleep(1)

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
        with self._monitor_lock:
            rows = list(self._cached_gpu_rows)
        self._render_gpu_process_table(rows)

    def _render_gpu_process_table(self, rows: list[tuple[str, str, str, str, str, str, str]]) -> None:
        table = self.query_one("#gpu_proc_table", DataTable)
        table.clear()
        if not rows:
            table.add_row("-", "-", "-", "-", "-", "-", "无 GPU 进程")
            return
        for row in rows:
            gpu_id, pid, user, sm_cell, cpu_cell, etime, command = row
            table.add_row(
                escape(gpu_id),
                escape(pid),
                escape(user),
                sm_cell,
                cpu_cell,
                escape(etime),
                escape(command),
            )

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
        clean_chars = []
        in_tag = False
        for ch in value:
            if ch == "[":
                in_tag = True
                continue
            if ch == "]":
                in_tag = False
                continue
            if not in_tag:
                clean_chars.append(ch)
        clean = "".join(clean_chars).replace("%", "").strip()
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
        selected_name = self._row_key_to_name.get(row_key, "")
        if selected_name:
            self.selected_instance = selected_name
            self._set_log_view(self.manager.read_log_tail(self.selected_instance, 60))

    @on(DataTable.RowHighlighted, "#instance_table")
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_key = str(event.row_key)
        highlighted_name = self._row_key_to_name.get(row_key, "")
        if highlighted_name:
            self.selected_instance = highlighted_name

    def _current_row_target(self) -> tuple[str, Optional[str]]:
        table = self.query_one("#instance_table", DataTable)
        row_idx = getattr(table, "cursor_row", None)
        if not isinstance(row_idx, int):
            return "none", None
        name = self._row_index_to_name.get(row_idx)
        if name == "__new__":
            return "new", None
        if name:
            return "existing", name
        return "none", None

    def _editable_column_key(self, col_idx: int) -> Optional[str]:
        mapping = {
            1: "name",
            2: "model",
            3: "port",
            4: "gpu_id",
            5: "conda_env",
            6: "weight_path",
        }
        return mapping.get(col_idx)

    def action_edit_cell(self) -> None:
        table = self.query_one("#instance_table", DataTable)
        row_type, name = self._current_row_target()
        col_idx = getattr(table, "cursor_column", None)
        if not isinstance(col_idx, int):
            self.post_status("请先将光标移动到可编辑列")
            return
        field = self._editable_column_key(col_idx)
        if not field:
            self.post_status("当前列不可编辑")
            return

        if row_type == "new":
            initial = self._draft_row.get(field, "")
        elif row_type == "existing" and name:
            initial = self._get_row_config(name).get(field, "")
        else:
            self.post_status("请先将光标移动到实例行")
            return

        title = f"编辑 {field}"
        self._monitor_paused = True
        if field in {"model", "conda_env"}:
            if field == "model":
                options = self.manager.get_registered_models()
                dialog_title = "选择模型（自动识别）"
            else:
                options = self.manager.get_conda_envs()
                dialog_title = "选择 Conda 环境（自动识别）"
            if not options:
                self._monitor_paused = False
                self.post_status(f"{field} 可选项为空，请先检查环境")
                return
            self.push_screen(
                ChoiceEditDialog(dialog_title, options, str(initial)),
                callback=lambda value: self._save_cell_edit(row_type, name, field, value),
            )
        else:
            self.push_screen(
                CellEditDialog(title, str(initial)),
                callback=lambda value: self._save_cell_edit(row_type, name, field, value),
            )

    def _save_cell_edit(self, row_type: str, name: Optional[str], field: str, value: Optional[str]) -> None:
        self._monitor_paused = False
        if value is None:
            self.post_status("已取消编辑")
            return
        value = value.strip()
        if row_type == "new":
            self._draft_row[field] = value
        elif row_type == "existing" and name:
            override = self._edited_rows.setdefault(name, {})
            override[field] = value
        self._touch_edit_lock(1.0)
        self.refresh_instances(force_table_refresh=True)

    def action_deploy(self) -> None:
        # 对空白行执行部署：填完最后一行后按 n 即可部署。
        model = self._draft_row["model"]
        instance_name = self._draft_row["name"]
        port = self._draft_row["port"] or str(self.manager.default_port)
        gpu_id = self._draft_row["gpu_id"] or str(self.manager.default_gpu_id)
        conda_env = self._draft_row["conda_env"]
        weight_path = self._draft_row["weight_path"]
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
        self._draft_row["name"] = ""
        self._draft_row["weight_path"] = ""
        self.refresh_instances(force_table_refresh=True)

    def _require_selection(self) -> Optional[str]:
        # 优先使用当前光标所在行，满足“光标到了就能操作”。
        row_type, name_from_cursor = self._current_row_target()
        if row_type == "existing" and name_from_cursor:
            if name_from_cursor != self.selected_instance:
                self.selected_instance = name_from_cursor
            return name_from_cursor
        if row_type == "new":
            self.post_status("当前是空白新实例行，请使用 n 部署")
            return None

        if self.selected_instance:
            return self.selected_instance
        self.post_status("请先将光标移动到左侧实例行")
        return None

    def action_stop_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._monitor_paused = True
        self.push_screen(
            ConfirmDialog("确认停止", f"确定要停止实例 `{name}` 吗？"),
            callback=lambda confirmed: self._do_stop_selected(name, confirmed),
        )

    def _do_stop_selected(self, name: str, confirmed: bool) -> None:
        self._monitor_paused = False
        if not confirmed:
            self.post_status("已取消停止")
            return
        _, msg = self.manager.stop_instance(name)
        self.post_status(msg)
        self.refresh_instances(force_table_refresh=True)

    def action_restart_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._monitor_paused = True
        self.push_screen(
            ConfirmDialog("确认重启", f"确定要重启实例 `{name}` 吗？"),
            callback=lambda confirmed: self._do_restart_selected(name, confirmed),
        )

    def _do_restart_selected(self, name: str, confirmed: bool) -> None:
        self._monitor_paused = False
        if not confirmed:
            self.post_status("已取消重启")
            return
        row_conf = self._get_row_config(name)
        model = row_conf["model"]
        instance_name = row_conf["name"] or name
        port = row_conf["port"] or str(self.manager.default_port)
        gpu_id = row_conf["gpu_id"] or str(self.manager.default_gpu_id)
        conda_env = row_conf["conda_env"]
        weight_path = row_conf["weight_path"]
        if not conda_env:
            self.post_status("重启失败：Conda 环境不能为空")
            return
        # 支持编辑后重启：允许实例名、端口、GPU 等参数被修改。
        self.manager.stop_instance(name)
        ok, msg = self.manager.start_instance(
            instance_name=instance_name or name,
            model_name=model,
            port=port,
            gpu_id=gpu_id,
            conda_env_name=conda_env,
            weight_path=weight_path,
            wait_for_ready=False,
        )
        self.selected_instance = instance_name or name
        self.post_status(msg if ok else f"重启失败: {msg}")
        self.refresh_instances(force_table_refresh=True)

    def action_delete_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._monitor_paused = True
        self.push_screen(
            ConfirmDialog("确认删除", f"确定要删除实例 `{name}` 吗？此操作不可恢复。"),
            callback=lambda confirmed: self._do_delete_selected(name, confirmed),
        )

    def _do_delete_selected(self, name: str, confirmed: bool) -> None:
        self._monitor_paused = False
        if not confirmed:
            self.post_status("已取消删除")
            return
        _, msg = self.manager.delete_instance(name)
        if self.selected_instance == name:
            self.selected_instance = ""
        self.post_status(msg)
        self.refresh_instances(force_table_refresh=True)

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
