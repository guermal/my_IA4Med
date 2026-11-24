import os
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from pathlib import Path
from trl import SFTTrainer
from datasets import Dataset as HFDataset
from datasat_loader import *


def to_hf_dataset(chest_xray_dataset):
    # Store flat, uniformly-typed string columns only. If we stored the nested
    # "messages" structure directly, Arrow would unify the schema of every content
    # dict in that column, bolting a spurious "image": None onto plain text dicts -
    # which then breaks qwen_vl_utils' image detection (it checks key presence, not
    # value). The real nested structure gets rebuilt fresh in build_collate_fn instead.
    rows = []
    for i in range(len(chest_xray_dataset)):
        messages = chest_xray_dataset[i]
        rows.append({
            "system_prompt": messages[0]["content"][0]["text"],
            "image_path": messages[1]["content"][0]["image"],
            "input_prompt": messages[1]["content"][1]["text"],
            "output_label": messages[2]["content"][0]["text"],
        })
    return HFDataset.from_list(rows)



def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text


def build_collate_fn(processor):
    processor.tokenizer.padding_side = "right"
    pad_token_id = processor.tokenizer.pad_token_id
    vision_token_ids = [
        processor.tokenizer.convert_tokens_to_ids(t)
        for t in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
    ]

    def collate_fn(examples):
        examples = [
            [
                {"role": "system", "content": [{"type": "text", "text": ex["system_prompt"]}]},
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image_path"]},
                    {"type": "text", "text": ex["input_prompt"]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["output_label"]}]},
            ]
            for ex in examples
        ]
        image_inputs = [process_vision_info(example)[0] for example in examples]

        full_texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
        batch = processor(text=full_texts, images=image_inputs, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        # Mask everything up to the assistant's response (system + user turns + generation
        # prompt) so loss is only computed on the target report tokens.
        prompt_texts = [
            processor.apply_chat_template(example[:-1], tokenize=False, add_generation_prompt=True)
            for example in examples
        ]
        prompt_batch = processor(text=prompt_texts, images=image_inputs, return_tensors="pt", padding=True)
        for i in range(len(examples)):
            prompt_len = int(prompt_batch["attention_mask"][i].sum())
            labels[i, :prompt_len] = -100

        labels[labels == pad_token_id] = -100
        for token_id in vision_token_ids:
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help=(
            "Attention implementation to use when loading the model. 'sdpa' (default) needs no extra "
            "install. 'flash_attention_2' is faster/more memory-efficient but requires the flash-attn "
            "package to be installed separately, which is nontrivial on Windows."
        ),
    )
    return parser.parse_args()










def main():
    args = parse_args()

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["model.visual"],  # keep the vision tower in bf16: small enough to afford, and its
                                        # feature dims don't align to bnb's fast-kernel blocksize anyway.
                                        # Must be "model.visual", not "visual" - Qwen2_5_VLForConditionalGeneration
                                        # nests it under .model.visual, and should_convert_module only matches
                                        # this list against the full dotted path (verified against
                                        # transformers.quantizers.quantizers_utils.should_convert_module directly).
    )

    # Load model and tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="qwen2.5-vl-3b-instruct-trl-sft-chestxray",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        max_length=None,
        # Our dataset rows are {"messages": [...]} with images nested inside, so TRL's
        # top-level "image"/"images" key heuristic can't auto-detect it as a vision dataset.
        # Skip TRL's own tokenization and rely entirely on our custom collate_fn.
        dataset_kwargs={"skip_prepare_dataset": True},
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=10,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_steps=0.03,  # Values < 1 are treated as a ratio of total steps for warmup
        remove_unused_columns=False,  # Keep the "messages" column; the model's forward() doesn't see it directly, our collate_fn does
        # Hub and reporting
        push_to_hub=True,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
    )
    import wandb

    wandb.init(
        project="qwen2.5-vl-3b-instruct-trl-sft-chestxray",
        name="qwen2.5-vl-3b-instruct-trl-sft-chestxray",
        config=training_args.to_dict(),
    )


    
    train_dataset = to_hf_dataset(ChestXrayDataset(csv_path=Path(r"D:\chestxray_dataset\train_dataset.csv")))
    eval_dataset = to_hf_dataset(ChestXrayDataset(csv_path=Path(r"D:\chestxray_dataset\test_dataset.csv"), mode='test'))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=build_collate_fn(processor),
    )
    trainer.train()

if __name__=='__main__':
    main()