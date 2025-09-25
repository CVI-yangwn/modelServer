import tornado
import tornado.ioloop
import tornado.web
from tornado.escape import json_decode, json_encode
import time
import asyncio
import sys
import os
import json
import uuid
from PIL import Image
from logger import set_logger, logger_output
from utils import save_cache_img
from models import *

_current_dir = os.path.abspath(os.path.dirname(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
SELECTED_MODEL_NAME = "Qwen3-14B" # Choose the model you want to deploy
SERVER_PORT = 5521
SERVER_ADDRESS = "0.0.0.0" # Listen on all interfaces

name_to_model_class = {
    "Qwen2.5-VL-3B": Qwen2_5_VL_3B,
    "Qwen2.5-VL-7B": Qwen2_5_VL_7B,
    "Qwen2.5-VL-32B": Qwen2_5_VL_32B,
    "Qwen3-14B": Qwen3_14B,
}

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

        await self.handle_non_stream_request(request_data)

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


# --- Model Inference Handler ---
class ChatCompletionHandler(BaseHandler):
    
    def process_non_stream(self, request_data: dict):
        """
        Processes the request using the loaded ML model.
        Input format is expected to be OpenAI-like.
        """
        messages = request_data.get("messages")

        try:
            user_message = messages[0] # Assuming the first message is the user's

            content_list = user_message.get("content", [])
            if not content_list:
                raise ValueError("Missing 'content' in user message.")

            images_path = []
            question = ""
            for i, content_item in enumerate(content_list):
                if content_item.get("type") == "text":
                    question = content_item.get("text")
                elif content_item.get("type") == "image_url":
                    save_img_path = save_cache_img(content_item['image_url']['url'], str(i), "temp")
                    images_path.append(save_img_path)

            if len(images_path) == 0:
                answer = self.model.ask_only_text(question)
            else:
                answer = self.model.ask_with_images(question, images_path)
            
            # 清理本地保存的图片
            for img_path in images_path:
                try:
                    os.remove(img_path)
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
        self.model_instance = name_to_model_class[SELECTED_MODEL_NAME].get_instance()
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
    if SELECTED_MODEL_NAME not in name_to_model_class:
        print(f"Error: Model '{SELECTED_MODEL_NAME}' not found in `name_to_model_class` mapping.")
        sys.exit(1)

    try:
        print("Starting Tornado server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        tornado.ioloop.IOLoop.current().stop()