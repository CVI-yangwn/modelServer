#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional

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
    app = MsTuiApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
