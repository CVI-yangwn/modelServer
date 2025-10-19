import urllib, urllib.request
from datetime import datetime
import base64, json
import os, ast, time
import numpy as np
import cv2
from logger import set_logger, logger_output

_current_dir = os.path.dirname(os.path.abspath(__file__))

SUPPORT_MODELS = ("Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "Qwen2.5-VL-32B",
                 "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
                 "Qwen3-14B", "Lingshu-7B", "Lingshu-32B", "Qwen3-VL-30B-A3B-Instruct", "HuatuoGPT-Vision-7B")

from openai import OpenAI, AsyncOpenAI
from collections import defaultdict

def set_no_proxy(url: str):
    if url.startswith("http://"):
        host = url.strip('http://').split(':')[0]
        os.environ["no_proxy"] = host
    else:
        pass

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/"

def post_local_sync(request_data: dict):
    with open(os.path.join(_current_dir, "config", "local_request.json"), "r") as f:
        model2url = json.load(f)
    model_name = request_data.get("model")
    try:
        url = model2url[model_name]
        set_no_proxy(url)

        client = OpenAI(api_key="no need", base_url=url, max_retries=0, timeout=1000)
        response = client.chat.completions.create(
            **request_data,
        )
        return response.choices[0].message.content
    except KeyError as e:
        logger = set_logger(f"{model_name}")
        logger.error(f"cann't find the key name {e} in local_request.json")
        raise ValueError("This is a mistake in config file, please contact the administrator~")
    except Exception as e:
        logger = set_logger(f"{model_name}")
        logger.error(f"Error in post_local_sync: {e}")
        raise ValueError(f"Error in post_local_sync: {e}")

def post_local_stream(request_data: dict):

    with open(os.path.join(_current_dir, "config", "local_request.json"), "r") as f:
        model2url = json.load(f)
    model_name = request_data.get("model")
    try:
        url = model2url[model_name]
        set_no_proxy(url)

        client = OpenAI(api_key='no need', base_url=url, max_retries=0, timeout=1000)
        response = client.chat.completions.create(
            **request_data,
        )
        answer = ''
        for chunk in response:
            answer += chunk.choices[0].delta.content
        return answer
    except Exception as e:
        logger = set_logger(model_name)
        logger.error(f"Error in post_local_stream: {e}")
        raise

def post_gemini_sync(request_data: dict, api_key: str):
    try:
        client = OpenAI(api_key=api_key, base_url=GEMINI_URL, max_retries=0, timeout=1000)
        response = client.chat.completions.create(
            **request_data,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger = set_logger("gemini")
        logger.error(f"Error in post_gemini_sync: {e}")
        raise ValueError("")

async def post_gemini_stream(request_data: dict, api_key: str):
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_URL, max_retries=0, timeout=1000)
        stream = await client.chat.completions.create(
            **request_data,
        )
        async for chunk in stream:
            yield chunk
    except Exception as e:
        logger = set_logger("gemini")
        logger.error(f"Error in post_gemini_stream: {e}")
        yield

def decode_base64_image(base64_string):
    """ 
    Decode a Base64 Data URI back into an image (NumPy array).
    
    Args:
        base64_string (str): Base64-encoded Data URI.
    
    Returns:
        np.ndarray: Decoded image as a NumPy array, or None if an error occurs.
    """
    try:
        # Split the Data URI into its components
        header, encoded = base64_string.split(',', 1)
        # Decode the Base64 string
        binary_data = base64.b64decode(encoded)
        # Convert binary data to a NumPy array
        np_array = np.frombuffer(binary_data, np.uint8)
        # Decode the image from the NumPy array
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image
    
    except Exception as e:
        print(f"Error: Decoding image - {e}")
        return None

def save_cache_img(image, name, check_type):
    image = decode_base64_image(image)
    
    current_file_path = os.path.dirname((os.path.abspath(__file__)))
    current_datetime = datetime.now()
    current_ymd = current_datetime.strftime("%Y-%m-%d")
    save_directory = os.path.join(current_file_path, f"cache_imgs/{current_ymd}/{check_type}")
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, f"{name}.jpg")
    if os.path.exists(save_path):
        counter = 1
        while True:
            file_name = f"{name}_{counter}.jpg"
            save_path = os.path.join(save_directory, file_name)
            if not os.path.exists(save_path):
                break
            counter += 1
    cv2.imwrite(save_path, image)
    return save_path