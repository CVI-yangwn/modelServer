import os
import importlib

_current_dir = os.path.abspath(os.path.dirname(__file__))

class ModelBase:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def ask_with_images(self, question:str, images:list, history=[]):
        """
        args:
            history: [
                {
                    content
                    role
                }
            ]
        """
        raise NotImplementedError

    def ask_only_text(self, question:str, history=[]):
        raise NotImplementedError

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
            self._instance: ModelBase = model_class(self.weight_path, **self.kwargs)
        
            self._test()
        return self._instance
    
    def _test(self):
        print("Testing...")
        try:
            self._instance.ask_only_text("Hello, world!")
            self._instance.ask_with_images("What is this?", [os.path.join(_current_dir, 'test_image.png')])
        except NotImplementedError:
            pass
        except Exception as e:
            print(f"Error during testing {self.class_name}: {e}")
            print("please check you have installed all packages for this model dependencies.")
            raise
        print(f"Test Successfully! {self.class_name} is ready to use.")
