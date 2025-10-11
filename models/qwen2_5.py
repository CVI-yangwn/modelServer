from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import os
import torch
from .model_base import ModelBase

class Qwen2_5_VL(ModelBase):
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    @torch.inference_mode()
    def generate(self, messages, **kwargs):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if kwargs.get("fps", None) is not None:
            video_kwargs["fps"] = kwargs["fps"]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            # **video_kwargs
        )
        
        # if inputs.input_ids.shape[1] > max_input_length:
        #     print("----------------------------")
        #     inputs.input_ids = inputs.input_ids[:, :max_input_length]
        #     if 'attention_mask' in inputs:
        #         inputs.attention_mask = inputs.attention_mask[:, :max_input_length]
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def ask_with_images(self, question:str, images:list, history=[]):
        """
        args:
            question: str, the question to ask
            images: list, a list of image paths
        """
        img_msg = [{"type": "image", "image": img} for img in images]
        messages = [
            {
                "role": "user",
                "content": img_msg + [{"type": "text", "text": question}],
            }
        ]
        if history:
            messages = messages + history
        return self.generate(messages)[0]
    
    def ask_with_videos(self, question:str, videos:list, **kwargs):
        """
        args:
            question: str, the question to ask
            videos: list, a list of video paths
            fps: float, the fps of the video, default is 3.0
            max_pixels: int, the max resolution of the video, default is 224*280
        """
        raise NotImplementedError("stupid thing")
        fps = kwargs.get("fps", 3.0)
        max_pixels = kwargs.get("max_pixels", 224*280)
        video_msg = [{"type": "video", "video": video, "fps": fps, "max_pixels": max_pixels} for video in videos]
        messages = [
            {
                "role": "user",
                "content": video_msg + [{"type": "text", "text": question}],
            }
        ]
        return self.generate(messages)[0]

    def ask_only_text(self, question:str, history=[]):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ]
        if history:
            messages = history + messages
        return self.generate(messages)[0]