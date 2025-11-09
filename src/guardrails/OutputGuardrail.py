from nltk.tokenize import sent_tokenize
from langdetect import detect, LangDetectException
import re, logging 

class OutputGuardrail:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = (logger or logging.getLogger("guardrails")).getChild("output")
    
    @staticmethod
    def _detect_citation_pattern(text: str, pattern: str) -> bool:
        matches = []
        for pat in pattern:
            matches.extend(re.findall(pat, text))
        if not matches: 
            return False
        return True
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"^[\'\"\(\[\s]+|[\'\"\)\]\s]+$", "", text)
    
    @staticmethod
    def _check_relevance(input: str, output: str) -> bool:
        input = OutputGuardrail._normalize_text(input)
        match_count = sum(tok.lower() in output.lower() for tok in input.split())
        ratio = match_count / len(input.split())
        return ratio > 0.60
    
    @staticmethod
    def _language_check(text: str, language: str) -> bool:
        try:
            return detect(text).lower().startswith(language.lower())
        except LangDetectException:
            return False

    @staticmethod
    def _validate_sent_length(text: str, min_sentences: int, max_sentences: int) -> bool:
        sentences = sent_tokenize(text)
        return len(sentences) >= min_sentences and len(sentences) <= max_sentences

    def check_completeness(self, input: str, output: str) -> bool:
        violations = []

        # 1. Check language constraints
        if not self._language_check(output, self.cfg["language"]):
            violations.append(f"Language mismatch: expected {self.cfg['language']}")
            self.logger.error("Language mismatch!")

        # 2. Check length constraints 
        if not self._validate_sent_length(output, self.cfg["min_sentences"],  self.cfg["max_sentences"]):
            n = len(sent_tokenize(output, language="english"))
            violations.append(f"Sentence count {n} outside [{self.cfg["min_sentences"]}, {self.cfg["max_sentences"]}]")
            self.logger.error("Sentence count violation!")

        # 3. Check citation presence
        if self.cfg["require_citations"]:
            if not self._detect_citation_pattern(output, self.cfg["citation_patterns"]):
                violations.append("Missing required citations in the output.")
                self.logger.error("Missing citations!")

        # 4. Check relevance
        if not self._check_relevance(input, output):
            violations.append("Output is not relevant to the input prompt.")
            self.logger.error("Relevance check failed!")   

        if not violations:
            return {"violations": None, "original_output": output}
        else:
            return {"violations": violations, "original_output": output}
    
