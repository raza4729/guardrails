from nltk.tokenize import sent_tokenize
from langdetect import detect_langs, detect
import re 

class OutputGuardrail:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def validate_schema(output: str, expected_format: str) -> bool:
        # Placeholder for format validation logic
        return True
    
    @staticmethod
    def _url_saftey_check(url: str) -> bool:
        # Placeholder for URL safety check logic
        return True  
    
    @staticmethod
    def _citation_check(text: str, pattern: str) -> bool:
        matches = []
        for pat in pattern:
            matches.extend(re.findall(pat, text))
        if not matches: 
            return False
        return True
    
    @staticmethod
    def _validate_relevance(text: str, context: str) -> bool:
        # Placeholder for relevance check logic
        return True
    
    @staticmethod
    def _language_check(text: str, language: str) -> bool:
        return detect(text).lower() == language.lower()

    @staticmethod
    def _validate_sent_length(text: str, min_sentences: int, max_sentences: int) -> bool:
        sentences = sent_tokenize(text)
        return len(sentences) >= min_sentences and len(sentences) <= max_sentences

    def check_completeness(self, output: str) -> bool:
        violations = []

        # 1. Check language constraints
        if not self._language_check(output, self.cfg["language"]):
            violations.append(f"Language mismatch: expected {self.cfg['language']}")

        # 2. Check length constraints 
        if not self._validate_sent_length(output, self.cfg["min_sentences"],  self.cfg["max_sentences"]):
            violations.append(f"Too many or few sentences: found {len(sent_tokenize(output))}, "
                            f"expected at least {self.cfg['min_sentences']}",
                            f" and at most {self.cfg['max_sentences']} sentences.")
        
        # 3. Check citation presence
        if self.cfg["require_citations"]:
            if not self._citation_check(output, self.cfg["citation_patterns"]):
                violations.append("Missing required citations in the output.")

        if not violations:
            return {"violations": None, "oiriginal_output": output}
        else:
            return {"violations": violations, "oiriginal_output": output}
    
