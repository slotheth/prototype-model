"""
Model loading, LoRA setup, and merging — shared by train, infer, and export.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

import config as cfg


def load_tokenizer(model_name: str | None = None) -> AutoTokenizer:
    """Load tokenizer with proper pad token setup."""
    name = model_name or cfg.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        name, token=cfg.HF_TOKEN, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str | None = None,
    quantize_4bit: bool = True,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """Load the base model with optional 4-bit quantization."""
    name = model_name or cfg.MODEL_NAME
    print(f"Loading model: {name} (4-bit={quantize_4bit})")

    kwargs = {
        "token": cfg.HF_TOKEN,
        "trust_remote_code": True,
        "device_map": device_map,
    }

    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def apply_lora(model) -> AutoModelForCausalLM:
    """Apply LoRA adapters to the model for training."""
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_trained_model(merge: bool = False):
    """Load base model with trained LoRA adapters.

    Args:
        merge: If True, merge LoRA weights into base model (for export).
               If False, keep as PeftModel (for inference with quantization).

    Returns:
        Tuple of (model, tokenizer).
    """
    if not cfg.LORA_DIR.exists():
        raise FileNotFoundError(
            f"LoRA adapters not found at {cfg.LORA_DIR}. Run training first."
        )

    if merge:
        # Load in fp16 on CPU for clean merge
        base = load_base_model(quantize_4bit=False, device_map="cpu")
    else:
        base = load_base_model(quantize_4bit=True)

    print(f"Loading LoRA from: {cfg.LORA_DIR}")
    model = PeftModel.from_pretrained(base, str(cfg.LORA_DIR))

    if merge:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    tokenizer = load_tokenizer()
    return model, tokenizer
