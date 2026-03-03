"""
llm.py
------
Loads and exposes the two LLMs used in the pipeline:
  - chat_llm    : google/medgemma-4b-it  (answer patient questions)
  - summ_llm    : deep-div/MediLlama-3.2 (summarise report chunks)
Both are loaded in 4-bit NF4 quantisation to fit on a single GPU.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# ---------------------------------------------------------------------------
# Shared quantisation config (used for both models)
# ---------------------------------------------------------------------------
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_model(model_id: str):
    """Return (tokenizer, model) for *model_id* with 4-bit quantisation."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bnb_config,
        device_map="auto",
    )
    return tokenizer, model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_chat_llm():
    """
    Load the chat / RAG model (medgemma-4b-it).

    Returns
    -------
    tuple[tokenizer, model, HF pipeline]
    """
    model_id = "google/medgemma-4b-it"
    tokenizer, model = _load_model(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        return_full_text=False,
        device=0,
    )
    return tokenizer, model, pipe


def load_summ_llm():
    """
    Load the summarisation model (MediLlama-3.2).

    Returns
    -------
    tuple[tokenizer, model, HF pipeline]
    """
    model_id = "google/medgemma-4b-it"
    tokenizer, model = _load_model(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
        device=0,
    )
    return tokenizer, model, pipe
