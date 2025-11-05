from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from llama_cpp import Llama
import logging, warnings

class Model(ABC):
    def __init__(self, model_id: str, cache: Any = None):
        self._model_id = model_id
        self.cache = cache or {}
        self._model = None

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def inference(self, prompt: str): ...

class Mistral(Model):
    def __init__(
            self, 
            model_id: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
            model_path: str = "mistral-7b-instruct-v0.2.Q2_K.gguf"):
        super().__init__(model_id=model_id)
        self._model_path = model_path

    def load_model(self) -> None:
        if self._model is None:
            self._model = Llama.from_pretrained(
                	repo_id= self._model_id,
	                filename=self._model_path,
                    verbose=False, # to suppress loading info
                )

    def inference(self, prompt: str) -> str:
        if prompt in self.cache:
            return self.cache[prompt]["response"]
        
        self.load_model()

        params = {
            "max_tokens": 120,
            "temperature": 0.2,
            "top_p": 0.95,
            "stop": ["</s>"],  # safer than "\n"
            "echo": False, # include prompt in output: False
        }

        output = self._model(prompt, **params)
        text = output["choices"][0]["text"] if output.get("choices") else ""
        self.cache[prompt] = {"role": "user", "response": text}
        return text 
    
class Claude(Model):

    def __init__(self, model_id, cache, api_key: str):
        super().__init__(model_id=model_id, cache=cache)
        self.api_key = api_key

    def load_model():
        # Implementation for loading Claude model
        pass

    def inference():
        # Implementation for Claude model inference
        pass

