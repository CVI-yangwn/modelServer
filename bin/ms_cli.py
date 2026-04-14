#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.ms_manager import MsManager


def print_status(instances):
    if not instances:
        print("当前没有托管任何实例。")
        return
    header = f"{'实例名':20} {'状态':10} {'PID':8} {'Conda':18} {'模型':24} {'端口':8} {'GPU':6}"
    print(header)
    print("-" * len(header))
    for item in instances:
        pid = str(item.pid) if item.pid else "N/A"
        print(
            f"{item.name:20} {item.state:10} {pid:8} {item.conda_env:18} {item.model:24} {item.port:8} {item.gpu_id:6}"
        )


def choose_conda_env(manager: MsManager) -> str:
    envs = manager.get_conda_envs()
    if not envs:
        raise RuntimeError("未发现可用 conda 环境")
    print("请选择 Conda 环境:")
    for idx, name in enumerate(envs, start=1):
        print(f"  {idx:2d}) {name}")
    while True:
        raw = input(f"输入序号 [1-{len(envs)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(envs):
            return envs[int(raw) - 1]
        print("输入无效，请重试。")


def main() -> int:
    manager = MsManager(PROJECT_DIR)
    parser = argparse.ArgumentParser(description="智能多实例服务管理")
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start")
    p_start.add_argument("instance")
    p_start.add_argument("-m", "--model", default=manager.default_model_name)
    p_start.add_argument("-p", "--port", default=manager.default_port)
    p_start.add_argument("-g", "--gpu", default=manager.default_gpu_id)
    p_start.add_argument("-e", "--env", default="")
    p_start.add_argument("-w", "--weight_path", default="")
    p_start.add_argument("--wait", action="store_true")

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("instance")

    p_restart = sub.add_parser("restart")
    p_restart.add_argument("instance")
    p_restart.add_argument("--wait", action="store_true")

    p_status = sub.add_parser("status")
    p_status.add_argument("instance", nargs="?")

    p_logs = sub.add_parser("logs")
    p_logs.add_argument("instance")
    p_logs.add_argument("-n", "--lines", type=int, default=80)

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("instance")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "start":
        conda_env = args.env or choose_conda_env(manager)
        ok, msg = manager.start_instance(
            instance_name=args.instance,
            model_name=args.model,
            port=args.port,
            gpu_id=args.gpu,
            conda_env_name=conda_env,
            weight_path=args.weight_path,
            wait_for_ready=args.wait,
        )
        print(msg)
        return 0 if ok else 1

    if args.command == "stop":
        ok, msg = manager.stop_instance(args.instance)
        print(msg)
        return 0 if ok else 1

    if args.command == "restart":
        ok, msg = manager.restart_instance(args.instance, wait_for_ready=args.wait)
        print(msg)
        return 0 if ok else 1

    if args.command == "status":
        if args.instance:
            item = manager.get_instance(args.instance)
            if not item:
                print(f"实例 {args.instance} 不存在")
                return 1
            print_status([item])
        else:
            print_status(manager.list_instances())
        return 0

    if args.command == "logs":
        print(manager.read_log_tail(args.instance, args.lines))
        return 0

    if args.command == "delete":
        confirm = input(f"确定删除实例 {args.instance}？[y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("已取消")
            return 0
        ok, msg = manager.delete_instance(args.instance)
        print(msg)
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
