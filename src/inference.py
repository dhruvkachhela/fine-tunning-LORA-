import argparse
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask_lora(question, schema, model, tokenizer, max_new_tokens=128):
    """
    Format the schema and question using the SFT chat template and generate a SQL query.
    """
    prompt = f"""Given the database schema:
{schema}

Write a SQL query to answer: {question}

Respond with only the SQL query. No explanation, no markdown formatting."""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False
        )
    
    # Extract assistant's response (excluding the prompt text)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Run SQL generation inference using LoRA fine-tuned Qwen2.5-0.5B.")
    parser.add_argument(
        "--question", 
        type=str, 
        default="What is the average salary of employees in the Engineering department?",
        help="Natural language question to translate into SQL."
    )
    parser.add_argument(
        "--schema", 
        type=str, 
        default="Table: employees (id, name, department, salary)",
        help="Database schema description or format."
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model identifier."
    )
    parser.add_argument(
        "--adapter_id", 
        type=str, 
        default="thefounder03/nl2sql-lora-qwen2.5-0.5b",
        help="LoRA adapter Hugging Face repository or local path."
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="Hugging Face access token (optional)."
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=128,
        help="Max new tokens to generate."
    )
    args = parser.parse_args()

    # Determine execution device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running inference on: {device}")
    
    # Read HF token from argument or environment variable
    hf_token = args.token or os.environ.get("HF_TOKEN")

    print(f"[Info] Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        token=hf_token
    )

    print(f"[Info] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        token=hf_token
    )

    print(f"[Info] Loading LoRA adapter: {args.adapter_id}")
    try:
        model = PeftModel.from_pretrained(
            base_model, 
            args.adapter_id, 
            token=hf_token
        )
    except Exception as e:
        print(f"[Error] Failed to load adapter from {args.adapter_id}. Error: {e}", file=sys.stderr)
        print("[Info] Falling back to using base model only.", file=sys.stderr)
        model = base_model

    print("\n--- Input Query ---")
    print(f"Schema:\n{args.schema}")
    print(f"Question: {args.question}")
    
    print("\nGenerating SQL...")
    sql_response = ask_lora(args.question, args.schema, model, tokenizer, max_new_tokens=args.max_tokens)
    
    print("\n--- Generated SQL Response ---")
    print(sql_response)
    print("------------------------------")

if __name__ == "__main__":
    import os
    main()
