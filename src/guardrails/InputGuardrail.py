from typing import Dict, Any
from langdetect import detect_langs, detect
import re, sys

class GuardrailViolation(Exception): pass

class InputGuardrail:
    def __init__(self, cfg, logger, model=None):
        self.cfg = cfg
        self.logger = logger.getChild("input")
        self.model = model

    @staticmethod
    def block_malicious_content(text: str, model) -> bool:
        if model is not None:
            prompt = "[INST] Classify the following user input as SAFE, TOXIC, or MALICIOUS. Output only the label.\n\n" + text + " [/INST]"
            response = model.inference(prompt)
            if "MALICIOUS" in response.upper():
                return True
            if "TOXIC" in response.upper():
                return True
            if "SAFE" in response.upper():
                return False
        raise ValueError("Model not provided for malicious content detection")
           
    @staticmethod
    def _language_check(text: str, language: str) -> bool:
        return detect(text).lower() == language.lower()
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"^[\'\"\(\[\s]+|[\'\"\)\]\s]+$", "", text)

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Combine instruction and text into a single prompt."""
        instruction = "Summarize the following abstract in maximum 3 sentences along side its factuality (such as citations, etc.)."
        
        abstract = input_data.get("abstract", "")
        abstract = self._normalize_text(abstract)

        if not abstract:
            self.logger.error("Abstract is empty") 
            raise GuardrailViolation("Abstract is empty")

        lang = self.cfg.get("language", "en")

        # 1. language check
        if not self._language_check(abstract, lang):
            self.logger.warning("Abstract language not supported", lang) 
            raise GuardrailViolation("Language not supported")

        # 2. malicious content check
        if self.block_malicious_content(abstract, self.model):
            self.logger.warning("Malicious content detected (len=%d)", len(abstract))  
            raise GuardrailViolation("Malicious content detected")

        return f"[INST] {instruction}\n\n{abstract} [/INST]"
