import tornado
import tornado.ioloop
import tornado.web
import time
import asyncio
import sys
import os
import shutil
import json
import uuid
from utils.logger import set_logger, logger_output
from utils.post_setting import save_cache_img
from models import *
import argparse

_current_dir = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5522)
    parser.add_argument('--addr', type=str, default="0.0.0.0")
    parser.add_argument('--model', type=str, default="Qwen2.5-VL-7B")
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=_current_dir)
    parser.add_argument('--thinking', type=bool, default=False)

    return parser.parse_args()

# --- Base Handler ---
class BaseHandler(tornado.web.RequestHandler):
    """
    Base handler for common functionalities.
    The `process` method should be overridden by child classes.
    """
    def initialize(self, model_instance, handler_name_for_log, executor):
        self.model: ModelBase = model_instance
        self.handler_name_for_log = handler_name_for_log
        self.logger = set_logger(name=self.handler_name_for_log) # Initialize logger once
        self.executor = executor

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()

    async def post(self):

        if self.executor._work_queue.qsize() >= 10:
            self.set_status(503) # Service Unavailable
            self.write({"error": "Server is busy, please try again later."})
            await self.finish()
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

        if model_name not in name_to_model_class:
            self.set_status(400)
            self.write({"error": f"Model '{model_name}' is not supported."})
            return

        if stream_mode:
            await self.handle_stream_request(request_data)
        else:
            await self.handle_non_stream_request(request_data)

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

    async def handle_non_stream_request(self, request_data: dict):
        model_name = request_data.get("model")

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

            self.set_header("Content-Type", "application/json") # 1. 先把正确的头设置好
            await self.flush()

            # --- 心跳保活循环 (这段逻辑保持不变) ---
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=1)
                except asyncio.TimeoutError:
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
            self.set_status(503)
            self.write(self.create_complete_response(f"Handle API request failed: {str(e)}", model_name))


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

import asyncio
from queue import Queue, Empty
# --- Model Inference Handler ---
class ChatCompletionHandler(BaseHandler):

    async def process_stream(self, request_data: dict):
        """
        Processes the request in a background thread and yields results via a queue.
        This simulates a streaming response for a non-streaming model.
        """
        
        # 1. 创建一个线程安全的队列，用于在同步和异步代码之间通信
        q = Queue()
        
        # 定义一个特殊的标记，表示流的结束
        STOP_SIGNAL = object()
        
        def run_model_in_thread():
            """
            这个函数将在后台线程中执行。
            它会调用阻塞的模型推理，然后将结果放入队列。
            """
            try:
                # 调用你已有的、同步的、阻塞的 process_non_stream 方法
                # 这是一种代码复用的好方法
                result_text = self.process_non_stream(request_data)
                q.put(result_text)
            except Exception as e:
                # 如果模型推理出错，将异常放入队列
                self.logger.error(f"Error during model inference in thread: {e}", exc_info=True)
                q.put(e)
            finally:
                # 无论成功还是失败，都放入结束信号
                q.put(STOP_SIGNAL)

        # 2. 在 executor 中异步运行上述函数
        # 我们不 await 它，因为它会立即返回一个 future 对象
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self.executor, run_model_in_thread)

        # 3. 异步地从队列中读取数据并 yield
        while True:
            try:
                # 非阻塞地从队列中获取项目
                # 我们不能用 q.get() 因为它是阻塞的
                item = q.get_nowait()
                
                if item is STOP_SIGNAL:
                    # 收到结束信号，跳出循环
                    break
                elif isinstance(item, Exception):
                    # 如果队列中的是异常对象，则重新抛出它
                    raise item
                else:
                    # 这是我们期待的文本结果
                    # 为了让它看起来像流，我们创建一个模拟的 chunk 对象
                    # 这个对象结构需要匹配你的 format_chunk_to_sse 方法的期望
                    from types import SimpleNamespace
                    
                    # 模拟 OpenAI 的 Stream Chunk 结构
                    mock_chunk = SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                index=0,
                                delta=SimpleNamespace(content=item), # 核心内容
                                finish_reason=None # 还未结束
                            )
                        ]
                    )
                    yield mock_chunk
                    
            except Empty:
                # 队列为空，说明后台线程还在计算
                # 我们等待一小段时间，让出控制权，避免阻塞事件循环
                await asyncio.sleep(0.01)

        # 等待后台线程任务最终完成，以便正确处理任何未捕获的异常
        await future
        
        # 4. 发送最后一个包含 finish_reason 的块
        from types import SimpleNamespace
        final_mock_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content=""), # 最后的 delta 内容为空
                    finish_reason="stop" # 标记流结束
                )
            ]
        )
        yield final_mock_chunk


    def process_non_stream(self, request_data: dict):
        """
        Processes the request using the loaded ML model.
        Input format is expected to be OpenAI-like.
        """
        messages = request_data.get("messages")

        try:
            # split messages
            history_messages: list = messages[:-1]  # =[] if have no history
            latest_user_message: dict = messages[-1]

            if latest_user_message.get("role") != "user":
                raise ValueError("The last message must be from the 'user'.")

            content_list = latest_user_message.get("content", [])
            if not content_list:
                raise ValueError("Missing 'content' in user message.")

            images_path = []
            question = ""

            unique_folder_name = str(uuid.uuid4())
            for i, content_item in enumerate(content_list):
                if content_item.get("type") == "text":
                    question = content_item.get("text")
                # add the latest images to message
                elif content_item.get("type") == "image_url":
                    save_img_path = save_cache_img(content_item['image_url']['url'], str(i), unique_folder_name)
                    images_path.append("file://"+save_img_path)

            # add historical images
            if history_messages:
                try:
                    for i, hm in enumerate(history_messages):
                        his_content_list = hm.get('content', [])
                        
                        for j, content_item in enumerate(his_content_list):
                            if content_item.get("type") == "image_url":
                                save_img_path = save_cache_img(content_item['image_url']['url'], f'h{i}_{j}', unique_folder_name)
                                content_item['image'] = "file://"+save_img_path
                                content_item['type'] = 'image'
                                content_item.pop('image_url')
                            # print(history_messages)
                except Exception as e:
                    self.logger.error("Failed to process historical images.")
                    raise e

            try:
                if len(images_path) == 0:
                    answer = self.model.ask_only_text(question, history=history_messages)
                else:
                    answer = self.model.ask_with_images(question, images_path, history=history_messages)
            except Exception as e:
                self.logger.error(f"Model inference error: {e}")
                raise e
            
            # 清理本地保存的图片
            for img_path in images_path:
                try:
                    parent_dir = os.path.dirname(img_path)
                    try:
                        shutil.rmtree(parent_dir)
                    except OSError:
                        pass
                except Exception as e:
                    # self.logger.warning(f"Failed to delete image {img_path}: {e}")
                    pass
            return answer

        except (KeyError, IndexError, TypeError, ValueError) as e:
            self.logger.error(f"Error infer: {e}")
            self.set_status(400)
            raise e  # 重新抛出异常，让上层处理


class ModelServer:
    def __init__(self, server_address="0.0.0.0", port=8000):
        self.model_instance = name_to_model_class[SELECTED_MODEL_NAME].get_instance(weight_path=WEIGHT_PATH)
        self.address = server_address
        self.port = port
        
    # --- Main Application Setup ---
    def build(self):
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=10)
        app = tornado.web.Application([
            (r"/v1/chat/completions", ChatCompletionHandler, dict(
                model_instance=self.model_instance,
                handler_name_for_log=SELECTED_MODEL_NAME,
                executor=executor
                )
            ),
        ])
        app.listen(self.port, address=self.address)
        logger = set_logger(name=SELECTED_MODEL_NAME)
        logger.info("API endpoint available at /local/chat/completions")


async def main():
    server = ModelServer(port=SERVER_PORT, server_address=SERVER_ADDRESS)
    server.build()
    await asyncio.Event().wait()


if __name__ == "__main__":
    args = parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    SELECTED_MODEL_NAME = args.model
    SERVER_PORT = args.port
    SERVER_ADDRESS = args.addr
    WEIGHT_PATH = args.weight_path

    if SELECTED_MODEL_NAME not in name_to_model_class:
        print(f"Error: Model '{SELECTED_MODEL_NAME}' not found in `name_to_model_class` mapping.")
        sys.exit(1)

    try:
        print("Starting Tornado server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        tornado.ioloop.IOLoop.current().stop()