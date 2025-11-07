from typing import Dict, Any
from langdetect import detect_langs, detect

class InputGuardrail:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def block_milicious_content(self, text: str) -> bool:
        # Placeholder for malicious content detection logic
        return False
    
    @staticmethod
    def _language_check(text: str, language: str) -> bool:
        return detect(text).lower() == language.lower()

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Combine instruction and text into a single prompt."""
        instruction = "Summarize the following abstract in maximum 3 sentences along side its factuality (such as citations, etc.)."
        
        abstract = input_data.get("abstract", "")
        abstract = abstract.lstrip("'([")
        abstract = abstract.rstrip("')")
        if not self._language_check(abstract, self.cfg["language"]):
            raise ValueError("Abstract language not supported")
        # Merge instruction and text
        combined = f"{instruction}\n\n{abstract}"

        # unsafe token cleaning
        safe_prompt = combined.replace("Ignore previous instructions", "[filtered]")
        return safe_prompt.strip()