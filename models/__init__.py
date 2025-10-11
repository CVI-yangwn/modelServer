__all__ = ['ModelBase', 'name_to_model_class']

import os
import importlib

_current_dir = os.path.abspath(os.path.dirname(__file__))

from .model_base import ModelBase
# from .qwen2_5 import Qwen2_5_VL
# from .qwen3 import Qwen3, Qwen3_VL
# from .deepseek import DeepSeek_VL

from .model_base import ModelBase

class LazyModel:
    def __init__(self, module_name: str, class_name: str, weight_path: str, **kwargs):

        self._instance = None
        self.module_name = module_name
        self.class_name = class_name
        self.weight_path = weight_path
        self.kwargs = kwargs

    def get_instance(self):
        if self._instance is None:
            try:
                module = importlib.import_module(self.module_name, package=__package__)
                
                # 从导入的模块中获取类对象
                model_class = getattr(module, self.class_name)
                
            except ImportError:
                print(f"Error: Cann't import '{self.module_name}'")
                print(f"Please check you have installed all packages for '{self.class_name}' model dependencies.")
                raise
            except AttributeError:
                print(f"Error: Cann't find class '{self.class_name}' in module '{self.module_name}'.")
                raise

            print(f"Creating {self.class_name} instance from {self.module_name}...")
            self._instance = model_class(self.weight_path, **self.kwargs)
        
        return self._instance

Qwen2_5_VL_3B = LazyModel('.qwen2_5', 'Qwen2_5_VL', os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-3B'))
Qwen2_5_VL_7B = LazyModel('.qwen2_5', 'Qwen2_5_VL', os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-7B'))
Qwen2_5_VL_32B = LazyModel('.qwen2_5', 'Qwen2_5_VL', os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-32B'))
Lingshu_7B = LazyModel('.qwen2_5', 'Qwen2_5_VL', "/mnt/shared_models/Lingshu-7B")
HuatuoGPT_Vision_7B = LazyModel('.qwen2_5', 'Qwen2_5_VL', "/mnt/shared_models/HuatuoGPT-Vision-7B")

Qwen3_14B = LazyModel('.qwen3', 'Qwen3', "/mnt/shared_models/Qwen3-14B", enable_thinking=True)
Qwen3_VL_30B_A3B_Instruct = LazyModel('.qwen3', 'Qwen3_VL', "/mnt/shared_models/Qwen3-VL-30B-A3B-Instruct")

DeepSeek_VL2 = LazyModel('.deepseek', 'DeepSeek_VL', "/mnt/shared_models/deepseek-vl2")

name_to_model_class = {
    "Qwen2.5-VL-3B": Qwen2_5_VL_3B,
    "Qwen2.5-VL-7B": Qwen2_5_VL_7B,
    "Qwen2.5-VL-32B": Qwen2_5_VL_32B,
    "Lingshu-7B": Lingshu_7B,
    "HuatuoGPT-Vision-7B": HuatuoGPT_Vision_7B,
    "Qwen3-14B": Qwen3_14B,
    "Qwen3-VL-30B-A3B-Instruct": Qwen3_VL_30B_A3B_Instruct,
    "DeepSeek-VL2": DeepSeek_VL2,
}