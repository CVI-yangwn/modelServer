"""
pip install git+https://github.com/microsoft/LLaVA-Med.git
"""
import torch
from .model_base import ModelBase
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class Llava_med_1_5(ModelBase):
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load_model(self):
        set_seed(0)
        # Model
        disable_torch_init()
        model_path = os.path.expanduser(self.model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)

    # def ask_with_images(self, question:str, images:list, history=[]):
    #     qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    #     if self.model.config.mm_use_im_start_end:
    #         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    #     # conv = conv_templates['vicuna_v1'].copy()
    #     conv = conv_templates['mistral_instruct'].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()

    #     input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    #     # image = Image.open(images[0])
        
    #     imgpaths = []
    #     for img_path in images:
    #         if img_path.startswith('file://'):
    #             imgpaths.append(img_path.split('file://')[1])
    #         else:
    #             imgpaths.append(img_path)
    #     image_tensor = process_images([Image.open(img_path) for img_path in imgpaths], self.image_processor, self.model.config)[0]

    #     with torch.inference_mode():
    #         output_ids = self.model.generate(
    #             input_ids,
    #             images=image_tensor.unsqueeze(0).half().cuda(),
    #             do_sample=False,
    #             temperature=0,
    #             top_p=None,
    #             num_beams=1,
    #             # no_repeat_ngram_size=3,
    #             max_new_tokens=1024,
    #             use_cache=True)

    #     outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    #     return outputs

    def ask_with_images(self, question:str, images:list, history=[]):
        # Parse history to extract previous conversations and images
        historical_images = []
        for msg in history:
            if msg["role"] == "user":
                content = msg["content"]
                for c in content:
                    if c["type"] == "image":
                        # Extract image path
                        image_path = c["image"].split("file://")[1] if c["image"].startswith('file://') else c["image"]
                        historical_images.append(image_path)
        
        # Prepare current question
        qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # Initialize conversation template
        conv = conv_templates['mistral_instruct'].copy()
        
        # Add history to conversation
        for msg in history:
            if msg["role"] == "user":
                # Extract text from user message
                user_text = ""
                for c in msg["content"]:
                    if c["type"] == "text":
                        user_text = c["text"].replace(DEFAULT_IMAGE_TOKEN, '')
                        break
                if user_text:
                    conv.append_message(conv.roles[0], user_text)
            elif msg["role"] == "assistant":
                # Extract text from assistant message
                assistant_text = ""
                for c in msg["content"]:
                    if c["type"] == "text":
                        assistant_text = c["text"]
                        break
                if assistant_text:
                    conv.append_message(conv.roles[1], assistant_text)
        
        # Add current question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Process all images (historical + current)
        imgpaths = []
        
        # Add historical images
        imgpaths.extend(historical_images)
        
        # Add current images
        for img_path in images:
            if img_path.startswith('file://'):
                imgpaths.append(img_path.split('file://')[1])
            else:
                imgpaths.append(img_path)
        
        # Load and process images
        if imgpaths:
            image_tensor = process_images([Image.open(img_path) for img_path in imgpaths], self.image_processor, self.model.config)[0].unsqueeze(0).half().cuda()
        else:
            image_tensor = None

        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=1024,
                    use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs
    
    def ask_only_text(self, question:str, history=[]):
        # Prepare current question
        qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        
        # Initialize conversation template
        conv = conv_templates['mistral_instruct'].copy()
        
        # Add history to conversation
        for msg in history:
            if msg["role"] == "user":
                # Extract text from user message
                user_text = ""
                for c in msg["content"]:
                    if c["type"] == "text":
                        user_text = c["text"].replace(DEFAULT_IMAGE_TOKEN, '')
                        break
                if user_text:
                    conv.append_message(conv.roles[0], user_text)
            elif msg["role"] == "assistant":
                # Extract text from assistant message
                assistant_text = ""
                for c in msg["content"]:
                    if c["type"] == "text":
                        assistant_text = c["text"]
                        break
                if assistant_text:
                    conv.append_message(conv.roles[1], assistant_text)
        
        # Add current question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = None
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=2048,
                    use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
