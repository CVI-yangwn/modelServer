"""
Based on trans_server.py
Transfer the remote api call to escape from timeout
"""
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
from openai import OpenAI
import asyncio
from queue import Queue, Empty

from logger import set_logger

os.environ['NO_PROXY'] = '*'  # forbit all the proxy 

def post_stream(request_data: dict):
    try:
        api_key = request_data.pop('api_key')
        base_url = request_data.pop('base_url')
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=1000)
        response = client.chat.completions.create(
            **request_data,
        )
        answer = ''
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content is not None:
                answer += delta.content
        return answer
    except Exception as e:
        logger = set_logger("tream")
        logger.error(f"Error in post_local_stream: {e}")
        raise

def post_sync(request_data: dict):
    try:
        api_key = request_data.pop('api_key')
        base_url = request_data.pop('base_url')
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=1000)
        response = client.chat.completions.create(
            **request_data,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger = set_logger("sync")
        logger.error(f"Error in post_gemini_sync: {e}")
        raise ValueError("")

class BaseHandler(tornado.web.RequestHandler):

    def initialize(self, executor: ThreadPoolExecutor, key_manager = None):
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
            task = asyncio.create_task(run_in_executor_and_get_result())

            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=3)
                except asyncio.TimeoutError:
                    self.write(" ")
                    await self.flush()
            
            if task.exception():
                error_message = str(task.exception()).replace('\n', ' ')
                error_payload = json.dumps({"error": f"Model inference failed: {error_message}"})
                self.write(f"data: {error_payload}\n\n")
                raise
                
            response_text = task.result()
            
            if not self._headers_written:
                 self.set_header("Content-Type", "application/json; charset=UTF-8")

            self.write(self.create_complete_response(response_text, model_name))
            
        except Exception as e:
            print(f"Error in non-stream request: {e}")
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
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "begin stream processing")
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

    async def process_stream(self, request_data: dict):
        """
        Processes the request in a background thread and yields results via a queue.
        This simulates a streaming response for a non-streaming model.
        """
        
        q = Queue()
        
        STOP_SIGNAL = object()
        
        def run_model_in_thread():
            """
            这个函数将在后台线程中执行。
            它会调用阻塞的模型推理，然后将结果放入队列。
            """
            try:
                result_text = post_stream(request_data)
                q.put(result_text)
            except Exception as e:
                # self.logger.error(f"Error during model inference in thread: {e}", exc_info=True)
                print(f"Error during model inference in thread: {e}")
                q.put(e)
            finally:
                q.put(STOP_SIGNAL)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self.executor, run_model_in_thread)

        while True:
            try:
                item = q.get_nowait()
                
                if item is STOP_SIGNAL:
                    break
                elif isinstance(item, Exception):
                    raise item
                else:
                    from types import SimpleNamespace
                    
                    mock_chunk = SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                index=0,
                                delta=SimpleNamespace(content=item),
                                finish_reason=None
                            )
                        ]
                    )
                    yield mock_chunk
                    
            except Empty:
                await asyncio.sleep(0.01)

        await future
        
        from types import SimpleNamespace
        final_mock_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content=""),
                    finish_reason="stop"
                )
            ]
        )
        yield final_mock_chunk

    def process_non_stream(self, request_data: dict):
        try:
                
            result = post_sync(request_data)
            
            return result

        except Exception as e:
            # self.set_status(503)
            # self.write({"error": f"Model call error for user {username}: {e}"})
            raise ValueError(f"Local inference error -- {e}")

class TransServer:
    def __init__(self, port, server_address):
        self.port = port
        self.address = server_address
        self.key_manager = None

    def build(self):
        executor = ThreadPoolExecutor(max_workers=80)
        app = tornado.web.Application([
            (r"/local/chat/completions", localHandler, dict(executor=executor, key_manager=self.key_manager)),
            (r"/v1/chat/completions", localHandler, dict(executor=executor, key_manager=self.key_manager))
        ])
        app.listen(self.port, address=self.address)
        logger = set_logger(name="trans_server")
        logger.info(f"Server starting on http://{self.address or '127.0.0.1'}:{self.port}")
        logger.info("API endpoint available at /v1/chat/completions")

async def main():
    server = TransServer(port=6666, server_address="0.0.0.0")
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