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
                 "Qwen3-14B")

from openai import OpenAI, AsyncOpenAI
from collections import defaultdict

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/"

def post_local_sync(request_data: dict):
    with open(os.path.join(_current_dir, "config", "local_request.json"), "r") as f:
        model2url = json.load(f)
    model_name = request_data.get("model")
    try:
        rul = model2url[model_name]

        client = OpenAI(api_key="no need", base_url=rul, max_retries=0, timeout=1000)
        response = client.chat.completions.create(
            **request_data,
        )
        return response.choices[0].message.content
    except KeyError as e:
        logger = set_logger(f"{model_name}")
        logger.error(f"cann't find the key name {e} in local_request.json")
        raise ValueError("This is a mistake in config file, please contact the administrator~")
    except:
        logger = set_logger(f"{model_name}")
        logger.error(f"Error in post_local_sync: {e}")
        raise ValueError("")

def post_gemini_sync(request_data: dict, api_key: str):
    try:
        client = OpenAI(api_key=api_key, base_url=GEMINI_URL, max_retries=0, timeout=300)
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
        client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_URL)
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

def check_image(check_type, body, logger):
    """
    The body of JSON should be like this:
    {
        "model": "Qwen2.5-VL-3B",
        "messages": [
                {"role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Please describe this image."
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": content,
                        }
                    }
                ],      
                "root&996": "test for build full platform"
            }
        ]
    } 
    """
    names = ["role", "content"]
    try:  # load json and check keys
        body_json = urllib.parse.parse_qs(body.decode('utf-8'))
        # === check support models
        if body_json['model'][0] not in MODEL_SUPPORT:
            return False, f"the model just only support: {MODEL_SUPPORT}"

        # body_json['messages'] is a list of string
        body_json['messages'] = ast.literal_eval(body_json['messages'][0])  # convert to json
        
        for n in names:
            if n not in body_json['messages'][0].keys():
                return False, f"missing key: {n}"
        try:  # check image for right format
            # content = ast.literal_eval(data["content"])  # convert to json
            content: list = body_json['messages'][0]['content']
            for dic in content:
                if dic['type'] == "image_url":
                    image = base64.b64decode(dic['image_url']['url'])
        except Exception as e:
            logger.error(f"Read img failed: {e}") 
            return False, f"read img failed"
        
        if "root&996" in body_json.keys():
            save_img_path = save_cache_img(image, body_json['root&996'][0], check_type)
            logger.debug(f"cache image in {save_img_path}")

        return True, "keys match success", body_json
    except:
        return False, "error: JSON parsing wrong"

def check_llama(check_type, body, logger):
    names = ["gender","age","main_compt","present_sym","west_dia",
             "face_dia","tongue_dia","pulse_dia","present_his",
             "past_his","person_his","marry_his","family_his"]
    try:  # load json and check keys
        logger.debug(f"body front 20: {body[:20]}, "
                     f"end 20: {body[-20:]}")
        body = urllib.parse.parse_qs(body.decode('utf-8'), keep_blank_values=True)
        data = {key: value[0] for key, value in body.items()}
        for n in names:
            if n not in data.keys():
                return False, f"missing key: {n}"
        logger.debug(f"input data:  {data}")
        return True, "keys match success", urllib.parse.urlencode(data).encode('utf-8')
    except:
        return False, "error: JSON parsing wrong"

check_data_functions = {
    "tongue": check_image,
    "face": check_image,
    "disease": check_llama,
    "syndrome": check_llama,
    "drugs": check_llama,
}

def check_args(check_type:str, body, logger):
    check_function = check_image
    return check_function(check_type, body, logger)

def request_for_model(url, params, timeout):
    try:
        request = urllib.request.Request(url, params)
        data = urllib.request.urlopen(request, timeout=timeout).read()
    except Exception as e:
        raise e
    return data

def post_qwen(params, timeout=100)->dict:
    # url = f"http://localhost:2559/{params['model'][0]}"
    url = f"http://localhost:2559/v1/chat/qwenvl"
    return request_for_model(url, params, timeout)

def post_gemini(model_name, messages, timeout=100):
    api_key = "AIzaSyCtsRd8ivbMx5Og934aMgLIWt-IlvcptEs"
    base_url = "https://generativelanguage.googleapis.com/v1beta/"
    client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
    print(" ==================== response")
    response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
    answer = response.choices[0].message.content
    print(" ===================get answer successfully")
    return answer

    # stream = client.chat.completions.create(
    #             model=model_name,
    #             messages=messages,
    #             stream=True
    #         )
    # for chunk in stream:
    #     content = chunk.choices[0].delta.content
    #     if content:
    #         print(content, end="", flush=True)
    # print() # 换行
    # print("streaming output...")
    # stream = client.chat.completions.create(
    #     model=model_name,
    #     messages=messages,
    #     stream=True
    # )
    # return stream
    # answer = "()"*1000
