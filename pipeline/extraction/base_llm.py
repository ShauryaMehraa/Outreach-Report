"""
Loads Ministral 3B once and exposes shared inference + JSON parsing helpers.
All extractor/generator classes inherit from this.
"""

import json
import logging
import re
from typing import Dict, List, Optional

import os
import torch
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModelForCausalLM

# Allow the model path to be overridden via environment variable so the same
# code works both on the shared JupyterHub server (global cache) and on a
# personal machine where the model is stored elsewhere.
_DEFAULT_MODEL_PATH = "/models/huggingface/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
MODEL_ID = os.getenv("LLM_MODEL_PATH", _DEFAULT_MODEL_PATH)
log = logging.getLogger(__name__)


class BaseLLM:

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading model on {self.device}...")

        token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=token,
        ).to(self.device)
        self.model.eval()
        log.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        messages: List[Dict],
        max_new_tokens: int = 800
    ) -> str:
        """Tokenize messages, generate, and return only the new decoded tokens."""
        # Guard against objects created via __new__ (weight-sharing pattern) that
        # somehow never had _share_model() called on them.
        if not hasattr(self, "model") or not hasattr(self, "tokenizer") or not hasattr(self, "device"):
            raise AttributeError(
                f"{type(self).__name__} was instantiated without model weights. "
                "Call _share_model(source, target) before using this instance."
            )
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.device)

        prompt_length = tokenized["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # JSON PARSING
    # ------------------------------------------------------------------

    def _safe_json(self, text: str, fallback: dict) -> dict:
        """Parse a JSON object from raw model output; return fallback on failure."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        log.warning("LLM JSON parse failed; using fallback schema. Raw output: %.200s", text)
        return fallback

    def _safe_parse_list(self, text: str) -> List[str]:
        """Parse a JSON array of strings from raw model output."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return []

    def _safe_parse_array(self, text: str) -> List[Dict]:
        """Parse a JSON array of objects from raw model output."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return []

    # ------------------------------------------------------------------
    # DEDUPLICATION
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(items: List[str], threshold: int = 85) -> List[str]:
        """Remove near-duplicate strings using fuzzy ratio."""
        unique = []
        for item in items:
            item = item.strip()
            if item and not any(fuzz.ratio(item, u) > threshold for u in unique):
                unique.append(item)
        return unique
