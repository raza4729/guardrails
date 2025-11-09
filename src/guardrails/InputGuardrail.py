from typing import Dict, Any
from langdetect import detect, LangDetectException
import re, sys, logging 

class GuardrailViolation(Exception): pass

class InputGuardrail:
    def __init__(self, cfg, logger, model=None):
        self.cfg = cfg
        self.logger = (logger or logging.getLogger("guardrails")).getChild("input")
        self.model = model

    def block_malicious_content(self, text: str) -> bool:
        if self.model is not None:
            prompt = "[INST] Classify the following user input as SAFE or MALICIOUS. Output only the label.\n\n" + text + " [/INST]"
            response = self.model.inference(prompt) or ""
            if "MALICIOUS" in response.upper():
                return True
            if "SAFE" in response.upper():
                return False
        raise False
           
    @staticmethod
    def _language_check(text: str, language: str) -> bool:
        try:
            return detect(text).lower().startswith(language.lower())
        except LangDetectException:
            return False
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"^[\'\"\(\[\s]+|[\'\"\)\]\s]+$", "", text)

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Combine instruction and text into a single prompt."""
        instruction = ("Summarize the following abstract in at most 3 sentences and include factual citations.")
 
        abstract = input_data.get("abstract", "")
        abstract_id = input_data.get("id", None)
        abstract = self._normalize_text(text=abstract)

        if not abstract:
            self.logger.error("Abstract is empty") 
            raise GuardrailViolation("Abstract is empty")

        lang = self.cfg.get("language", "en")

        # 1. language check
        if not self._language_check(text=abstract, language=lang):
            self.logger.warning("Language not supported", extra={"expected_lang": lang, "id": abstract_id})
            raise GuardrailViolation("Language not supported")

        # 2. malicious content check
        if self.block_malicious_content(text=abstract):
            self.logger.warning("Malicious content detected", extra={"id": abstract_id})  
            raise GuardrailViolation("Malicious content detected")

        return f"[INST] {instruction}\n\n{abstract} [/INST]"
