"""
pipeline.py — Orchestrates the guardrails and model inference flow.
"""

from src.models import Mistral
from .guardrails.InputGuardrail import  InputGuardrail, GuardrailViolation
from .guardrails.OutputGuardrail import OutputGuardrail

from typing import Dict, Any
import json, sys
from pathlib import Path
import kagglehub
import csv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("guardrails.pipeline")

class GuardrailPipeline:
    def __init__(self, model=None):
        # allow model injection for testing
        self.model = model or Mistral()
        self.cfg = {}

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full guardrail pipeline on one test or user input."""

        # load configs 
        with open(r"src\\guardrails\\guardrails_config.json") as f:
            self.cfg = json.load(f)["constraints"]
        
        # Input Guardrails
        inGuardrail_instance = InputGuardrail(cfg=self.cfg, logger=logger, model=self.model)
        try: 
            prompt = inGuardrail_instance.build_prompt(input_data=input)
        except GuardrailViolation as e:
            logger.warning("Input blocked: %s", e)
            return {
                "id": input.get("id", None),
                "task": input.get("abstract", None),
                "Skipped": True,
                "violations": [str(e)],
            }    
        
        # Model Inference
        try:
            output_text = self.model.inference(prompt)
        except Exception as e:
            return {"error": f"Model error: {e}"}
        
        # Output Guardrails 
        outGuardrail_instance = OutputGuardrail(cfg=self.cfg, logger=logger)
        results = outGuardrail_instance.check_completeness(input=input["abstract"], output=output_text)

        return {
            "id": input.get("id", None),
            "task": input.get("abstract", None),
            "input_prompt": prompt,
            "original_output": results.get("original_output", None),
            "violations": results.get("violations", None),
        }


# ---- entry point ---- #
if __name__ == "__main__":
    
    # load the pubmed dataset from kaggle
    pubmed_data = []
    path = kagglehub.dataset_download("bonhart/pubmed-abstracts")

    pipe = GuardrailPipeline()

    with open(path + "\\pubmed_abstracts.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        counter = 0
        for row in reader:
            line = {"abstract": row["deep_learning"], "citation": row["deep_learning_links"]}
            result = pipe.run(line)
            print(json.dumps(result, ensure_ascii=False))
            counter += 1
            if counter >= 5:
                break
