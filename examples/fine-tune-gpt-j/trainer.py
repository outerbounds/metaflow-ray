import os
import torch
import evaluate
import numpy as np
from ray.air import session
from typing import Dict, Any
from dataclasses import dataclass

from transformers import (
    Trainer,
    TrainingArguments,
    GPTJForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from transformers.utils.logging import disable_progress_bar, enable_progress_bar


MODEL_NAME = "EleutherAI/gpt-j-6B"
MAX_RETRIES = 5
CACHE_DIR = ".cache/GPTJ-6B"


@dataclass
class TrainerConfig:
    batch_size: int = 4
    epochs: int = 2
    warmup_steps: int = 0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


def get_deepspeed_config() -> Dict[str, Any]:
    return {
        "fp16": {
            "enabled": "auto",
            "initial_scale_power": 8,
        },
        "bf16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_model_with_retry(model_name: str) -> GPTJForCausalLM:
    for retry in range(MAX_RETRIES):
        try:
            model = GPTJForCausalLM.from_pretrained(
                model_name,
                use_cache=True,
                resume_download=True,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
            )
            return model
        except Exception as e:
            print(f"Attempt {retry + 1}/{MAX_RETRIES} failed with error: {e}")
            if retry < MAX_RETRIES - 1:
                print("Retrying...")
            else:
                raise Exception("Max retries reached. Failed to load model.")


def trainer_init_per_worker(
    train_dataset,
    eval_dataset=None,
    **config
) -> Trainer:
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )
    torch.backends.cuda.matmul.allow_tf32 = True

    trainer_config = TrainerConfig(**config)

    training_args = TrainingArguments(
        "output",
        per_device_train_batch_size=trainer_config.batch_size,
        per_device_eval_batch_size=trainer_config.batch_size,
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        warmup_steps=trainer_config.warmup_steps,
        num_train_epochs=trainer_config.epochs,
        logging_steps=1,
        save_strategy="no",
        label_names=["input_ids", "attention_mask"],
        push_to_hub=False,
        disable_tqdm=True,
        fp16=True,
        gradient_checkpointing=True,
        deepspeed=get_deepspeed_config(),
    )

    disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_retry(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    enable_progress_bar()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
