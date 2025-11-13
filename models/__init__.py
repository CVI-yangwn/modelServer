__all__ = ['ModelBase', 'name_to_model_class']

import os

_current_dir = os.path.abspath(os.path.dirname(__file__))

from .model_base import ModelBase, LazyModel

Qwen2_5_VL_3B = LazyModel('.qwen2_5', 'Qwen2_5_VL', os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-3B'))
Qwen2_5_VL_7B = LazyModel('.qwen2_5', 'Qwen2_5_VL', os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-7B'))
Qwen2_5_VL_32B = LazyModel('.qwen2_5', 'Qwen2_5_VL', os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-32B'))
Lingshu_7B = LazyModel('.qwen2_5', 'Qwen2_5_VL', "/mnt/shared_models/Lingshu-7B")
HuatuoGPT_Vision_7B = LazyModel('.qwen2_5', 'Qwen2_5_VL', "/mnt/shared_models/HuatuoGPT-Vision-7B")

VideoLLaVaNext7B = LazyModel('.qwen2_5', 'VideoLLaVa', "/mnt/shared_models/LLaVA-NeXT-Video-7B-hf")

Qwen3_14B = LazyModel('.qwen3', 'Qwen3', "/mnt/shared_models/Qwen3-14B", enable_thinking=True)
Qwen3_VL_30B_A3B_Instruct = LazyModel('.qwen3', 'Qwen3_VL_Moe', "/mnt/shared_models/Qwen3-VL-30B-A3B-Instruct")
Qwen3_VL_8B_Instruct = LazyModel('.qwen3', 'Qwen3_VL', "/mnt/shared_models/Qwen3-VL-8B-Instruct")

medgemma_27B = LazyModel('.qwen3', 'Qwen3_VL', "/mnt/shared_models/medgemma-27b-it")

DeepSeek_VL2 = LazyModel('.deepseek', 'DeepSeek_VL', "/mnt/shared_models/deepseek-vl2")

Llava_med_1_5 = LazyModel('.llava', 'Llava_med_1_5', "/mnt/shared_models/llava-med-v1.5-mistral-7b")

name_to_model_class = {
    "Qwen2.5-VL-3B": Qwen2_5_VL_3B,
    "Qwen2.5-VL-7B": Qwen2_5_VL_7B,
    "Qwen2.5-VL-32B": Qwen2_5_VL_32B,
    "Lingshu-7B": Lingshu_7B,
    "HuatuoGPT-Vision-7B": HuatuoGPT_Vision_7B,
    "Qwen3-14B": Qwen3_14B,
    "Qwen3-VL-30B-A3B-Instruct": Qwen3_VL_30B_A3B_Instruct,
    "Qwen3-VL-8B": Qwen3_VL_8B_Instruct,
    "DeepSeek-VL2": DeepSeek_VL2,
    "Llava-med-1.5": Llava_med_1_5,
    "medgemma-27B": medgemma_27B,
    "LLaVA-NeXT-Video-7B": VideoLLaVaNext7B,
}