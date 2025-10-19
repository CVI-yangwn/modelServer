from transformers import AutoModelForCausalLM, AutoTokenizer
from .model_base import ModelBase

class Qwen3(ModelBase):
    def __init__(self, model_path, enable_thinking=True):
        super().__init__(model_path)
        self.enable_thinking = enable_thinking

    def _load_model(self):
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    def ask_only_text(self, question:str, history=[]):
        # prepare the model input
        messages = [
            {"role": "user", "content": question}
        ]
        if history:
            messages = history + messages
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # the result will begin with thinking content in <think></think> tags, followed by the actual response
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
    
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from .qwen2_5 import Qwen2_5_VL
class Qwen3_VL_Moe(Qwen2_5_VL):
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load_model(self):
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(self.model_path, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

from transformers import AutoModelForImageTextToText, AutoProcessor

class Qwen3_VL(Qwen2_5_VL):
    def __init__(self, model_path):
        super().__init__(model_path)
    def _load_model(self):

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, dtype="auto", device_map="auto"
        )