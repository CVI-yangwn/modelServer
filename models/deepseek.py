import torch
from transformers import AutoModelForCausalLM, DeepseekV2ForCausalLM
from .model_base import ModelBase
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images

class DeepSeek_VL(ModelBase):
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer

    @torch.inference_mode()
    def generate(self, messages, **kwargs):
        pil_images = load_pil_images(messages)

        prepare_inputs = self.processor(
            conversations=messages,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to("cuda")

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language.forward(
            input_ids=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            # max_length=1024
            # pad_token_id=self.tokenizer.eos_token_id,
            # bos_token_id=self.tokenizer.bos_token_id,
            # eos_token_id=self.tokenizer.eos_token_id,
            # max_new_tokens=1024,
            # do_sample=False,
            # use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer

    def ask_with_images(self, question: str, images: list):
        # multiple images/interleaved image-text
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n"*len(images) + question,
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        return self.generate(conversation)




