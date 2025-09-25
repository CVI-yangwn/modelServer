import os

class ModelBase:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def ask_with_images(self, question:str, images:list):
        raise NotImplementedError

    def ask_only_text(self, question:str):
        raise NotImplementedError