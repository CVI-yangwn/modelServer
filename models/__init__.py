"""
"Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "Qwen2.5-VL-32B",
"gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"
"""

__all__ = ['ModelBase', 'Qwen2_5_VL_3B', 'Qwen2_5_VL_7B', 'Qwen2_5_VL_32B',
           'Qwen3_14B']

import os
_current_dir = os.path.abspath(os.path.dirname(__file__))

from .model_base import ModelBase
from .qwen2_5 import Qwen2_5_VL
from .qwen3 import Qwen3

class LazyModel:
    def __init__(self, model_class:ModelBase, weight_path, **kwargs):
        self._instance = None
        self.model_class = model_class
        self.weight_path = weight_path
        self.kwargs = kwargs

    def get_instance(self):
        if self._instance is None:
            print(f"Creating {self.model_class} instance...")
            self._instance = self.model_class(self.weight_path, **self.kwargs)
        return self._instance
    
Qwen2_5_VL_3B = LazyModel(Qwen2_5_VL, os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-3B'))
Qwen2_5_VL_7B = LazyModel(Qwen2_5_VL, os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-7B'))
Qwen2_5_VL_32B = LazyModel(Qwen2_5_VL, os.path.join(_current_dir, 'weights', 'Qwen2.5-VL-32B'))
Qwen3_14B = LazyModel(Qwen3, os.path.join(_current_dir, 'weights', 'Qwen3-14B'), enable_thinking=True)