# -*- coding: utf-8 -*-
"""
Fine-tuning Qwen2.5-0.5B-Instruct on the Spider text-to-SQL benchmark.
This script has been cleaned and structured to run locally or in a remote VM.
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def ask(question, schema, model, tokenizer, max_new_tokens=128):
    """
    Generate a SQL query for a given question and database schema using the model.
    """
    prompt = f"""Given the database schema:
{schema}

Write a SQL query to answer: {question}"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def format_schema(db_id, schema_lookup):
    """
    Retrieve and format schema information for a given db_id.
    """
    row = schema_lookup[db_id]
    formatted = f"Schema:\n{row['Schema (values (type))']}"
    if row["Foreign Keys"]:
        formatted += f"\nForeign Keys:\n{row['Foreign Keys']}"
    return formatted

def build_prompt(question, schema_text):
    """
    Construct the final instruction prompt containing the schema and question.
    """
    return f"""Given the database schema:
{schema_text}

Write a SQL query to answer: {question}

Respond with only the SQL query. No explanation, no markdown formatting."""

def format_example(row, schema_lookup):
    """
    Format a dataset row into standard ChatML user/assistant conversation structure.
    """
    schema_text = format_schema(row["db_id"], schema_lookup)
    prompt = build_prompt(row["question"], schema_text)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["query"]}
        ]
    }

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Load tokenizer and base model
    print(f"Loading base model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    # Simple sanity check query before fine-tuning
    print("\n--- Running Baseline Query Test ---")
    test_schema = "Table: employees (id, name, department, salary)"
    test_question = "What is the average salary of employees in the Engineering department?"
    print(f"Question: {test_question}")
    baseline_response = ask(test_question, test_schema, model, tokenizer)
    print(f"Baseline SQL Response:\n{baseline_response}\n")

    # Load Spider datasets
    print("Loading Spider dataset and Richardr1126/spider-schema...")
    dataset = load_dataset("xlangai/spider")
    schema_dataset = load_dataset("richardr1126/spider-schema")
    schema_lookup = {row["db_id"]: row for row in schema_dataset["train"]}

    # Prepare datasets
    print("Formatting and mapping training/validation splits...")
    formatted_dataset = dataset["train"].map(
        lambda r: format_example(r, schema_lookup),
        remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["validation"].select(range(200)).map(
        lambda r: format_example(r, schema_lookup),
        remove_columns=dataset["validation"].column_names
    )

    # Apply tokenizer chat templates
    def apply_template(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}

    formatted_dataset = formatted_dataset.map(apply_template)
    eval_dataset = eval_dataset.map(apply_template)

    print(f"Sample formatted training example text:\n{formatted_dataset[0]['text']}\n")

    # Configure LoRA
    print("Configuring LoRA Adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args for full training run
    print("Configuring SFTTrainer settings...")
    training_args = SFTConfig(
        output_dir="./nl2sql-lora-full",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        max_seq_length=1024,
        fp16=(device == "cuda"),
        bf16=False,
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        eval_dataset=eval_dataset,
    )

    # Run SFT training
    print("Starting SFT Training...")
    try:
        trainer.train()
        print("Training completed successfully.")
    except KeyboardInterrupt:
        print("Training interrupted by user. Stopping early.")

if __name__ == "__main__":
    main()