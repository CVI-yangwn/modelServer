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

`ms` 是一个智能多实例服务管理脚本，可以方便地启动、停止、查看和管理多个 ModelServer 实例。

### 注意
这个脚本实际上调用了代码modelServer.py，里面的一些模型的路径需要到models/\_\_init\_\_.py中进行自定义路径（用什么改什么就行），应该是比较简单的方式了。

### 快速开始

```bash
# 进入 bin 目录
cd bin

# 启动服务（会交互式选择 Conda 环境）
./ms start <实例名> -m <模型名> -p <端口> -g <GPU_ID>

# 或者从项目根目录直接调用
./bin/ms start <实例名> -m <模型名> -p <端口> -g <GPU_ID>

# 查看所有实例状态
./ms status

# 查看指定实例状态
./ms status <实例名>

# 查看日志
./ms logs <实例名>

# 停止服务
./ms stop <实例名>

# 重启服务
./ms restart <实例名>

# 删除实例（包括配置和日志）
./ms delete <实例名>
```

### 命令详解

#### 1. 启动服务 (`start`)

启动一个新的服务实例。启动时会交互式选择 Conda 环境。

```bash
./ms start <实例名> [选项]
```

**选项：**
- `-m, --model <模型名>`: 指定要加载的模型（默认: `Qwen2.5-VL-7B`）
- `-p, --port <端口号>`: 指定服务监听的端口（默认: `9960`）
- `-g, --gpu <GPU_ID>`: 指定使用的 GPU ID（默认: `0`）
- `-w, --weight_path <路径>`: 指定模型权重路径（可选，默认: 空）

**示例：**

```bash
# 启动一个名为 'api_server_1' 的实例，使用 GPU 1，端口 8001，模型 Qwen3-14B
./ms start api_server_1 -m Qwen3-14B -g 1 -p 8001

# 启动时指定模型权重路径
./ms start api_server_1 -m Qwen3-14B -g 1 -p 8001 -w /path/to/model/weights

# 使用默认参数启动
./ms start my_server
```

**启动流程：**
1. 脚本会自动检测可用的 Conda 环境
2. 交互式选择要使用的 Conda 环境
3. 在后台启动服务并等待初始化完成（最长 30 分钟）
4. 脚本会监控日志，当检测到 "Test Successfully!" 时认为启动成功
5. 启动成功后会在 `logs/<实例名>/` 目录下保存配置、PID 和日志文件

#### 2. 查看状态 (`status`)

查看服务实例的运行状态。

```bash
# 查看所有实例状态
./ms status

# 查看指定实例状态
./ms status <实例名>
```

**输出信息包括：**
- 实例名
- 运行状态（运行中/已停止）
- PID
- Conda 环境
- 模型名称
- 端口号
- GPU ID

**注意：** 查看单个实例状态时，还会显示权重路径（如果配置了的话）

#### 3. 查看日志 (`logs`)

实时查看指定实例的日志输出。

```bash
./ms logs <实例名>
```

按 `CTRL+C` 停止查看日志。

#### 4. 停止服务 (`stop`)

停止一个正在运行的服务实例。

```bash
./ms stop <实例名>
```

#### 5. 重启服务 (`restart`)

重启一个服务实例，会使用原有的配置（包括 Conda 环境、模型、端口、GPU、权重路径等）。

```bash
./ms restart <实例名>
```

**注意：** 重启时会自动使用之前保存的所有配置参数，无需重新指定。

#### 6. 删除实例 (`delete`)

停止并彻底删除一个实例的所有数据（包括配置和日志）。

```bash
./ms delete <实例名>
```

**注意：** 此操作会要求确认，删除后无法恢复。

### 使用示例

```bash
# 1. 启动第一个服务实例
./ms start qwen_server_1 -m Qwen2.5-VL-7B -p 8001 -g 0

# 2. 启动第二个服务实例（使用不同的 GPU 和端口，并指定权重路径）
./ms start qwen_server_2 -m Qwen3-14B -p 8002 -g 1 -w /path/to/weights

# 3. 查看所有实例状态
./ms status

# 4. 查看第一个实例的日志
./ms logs qwen_server_1

# 5. 重启第一个实例
./ms restart qwen_server_1

# 6. 停止第二个实例
./ms stop qwen_server_2

# 7. 删除第一个实例
./ms delete qwen_server_1
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