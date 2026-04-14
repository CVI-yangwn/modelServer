# 声明
这份代码主要是为了方便大模型做zero shot的测试，可以类似用OpenAI一样调用API，在本地部署了一个model server，里面的推理过程都可以自定义，方便很多。

因为本人一些其他的需求，所以有的代码是多余用不上的，看看modelServer就行。

## 支持模型

| 模型名称 | 类型 | 适配 |
|---------|------|------|
| Qwen2.5-VL Series | VLM | ✓ |
| Qwen3-14B | LLM | ✓ |
| Qwen3-VL Series | VLM | ✓ |
| Qwen3-VL-30B-A3B-Instruct | VLM | ✓ |
| Lingshu | VLM | ✓ |
| HuatuoGPT-Vision-7B | VLM | ✓ |
| DeepSeek-VL2 | VLM | ✓ |
| Llava-med-1.5 | VLM | ✓ |
| medgemma | VLM | ✓ |
| LLaVA-NeXT-Video-7B | VLM | ✓ |
| InternVL3 Series | VLM | ✓ |
| InternVL3.5 Series | VLM | ✓ |
| Llama3-VILA Series | VLM |  |

**注意：** 模型路径需要在 `models/__init__.py` 中配置，使用 `-w` 参数可以指定自定义权重路径。

# ModelServer 使用说明

## 使用 ms 脚本管理服务

`ms` 现在默认进入 Textual TUI（终端界面），用于多实例部署、编辑和监控。

### 依赖

TUI 依赖 `textual` 和 `pynvml`（强制）：

```bash
pip install -r requirements.txt
```

### 启动方式

```bash
# 在 modelServer 目录
./bin/ms
```

### TUI 主要功能

- 左侧是实例表（同时也是配置表），字段包括：
  - 实例名、模型、端口、GPU、Conda、路径、状态、PID
- 实例表最后一行固定为 `new` 空白行，可直接填写后部署
- 下方有实时系统监控（1s 刷新）和 GPU 进程信息
- 右侧显示当前选中实例日志

### 表格编辑与部署

- 将光标移动到目标单元格，按 `e` 编辑
- `model` 与 `conda` 字段使用自动识别候选（选择框）：
  - 模型来源：`models/__init__.py` 中 `name_to_model_class`
  - conda 来源：本机 `conda env list`
- 编辑空白 `new` 行后，按 `n` 部署新实例
- 编辑已有实例行后，按 `r` 可按编辑后的配置重启

### 快捷键

- `e`: 编辑当前单元格
- `n`: 部署空白新实例行
- `s`: 停止实例（含二次确认）
- `r`: 重启实例（含二次确认）
- `d`: 删除实例（含二次确认）
- `l`: 日志跟随开关
- `q`: 退出

### 命令行模式（兼容保留）

仍支持原有命令行用法（带参数时走 CLI）：

```bash
./bin/ms start <实例名> -m <模型名> -p <端口> -g <GPU_ID> -e <conda_env> [-w <weight_path>]
./bin/ms stop <实例名>
./bin/ms restart <实例名>
./bin/ms status [实例名]
./bin/ms logs <实例名>
./bin/ms delete <实例名>
```

### 文件结构

启动服务后，会在 `logs/` 目录下为每个实例创建独立的目录：

```
logs/
  └── <实例名>/
      ├── <实例名>.conf    # 配置文件（模型、端口、GPU、Conda环境、权重路径等）
      ├── <实例名>.pid     # 进程 ID 文件
      └── <实例名>.log     # 日志文件
```

### 注意事项

1. **Conda 环境**：启动服务时会交互式选择 Conda 环境，确保已安装所需的依赖
2. **端口冲突**：确保不同实例使用不同的端口号
3. **GPU 资源**：确保指定的 GPU ID 可用，避免多个实例使用同一 GPU 导致资源竞争
4. **启动超时**：服务初始化最长等待时间为 30 分钟，如果超时请检查日志
5. **启动成功判断**：脚本通过检测日志中的 "Test Successfully!" 来判断服务是否成功启动
6. **权重路径**：如果模型需要指定权重路径，使用 `-w` 或 `--weight_path` 参数；权重路径会保存在配置文件中，重启时会自动使用
7. **环境变量**：脚本会自动设置 `CUDA_VISIBLE_DEVICES` 和 `OMP_NUM_THREADS=8` 环境变量
8. **工作目录**：脚本会自动切换到项目根目录（modelServer 目录）执行
9. **日志位置**：所有日志保存在 `logs/<实例名>/<实例名>.log`

### 帮助信息

查看完整的帮助信息：

```bash
./ms --help
# 或
./ms -h
```

# TODO
- [ ] 传入推理参数
- [ ] 整理适配其他模型的环境
- [ ] 读图的方式是先保存再读，可能有点粗糙，但一开始是为了更好适应不同模型，后面可以针对性修改