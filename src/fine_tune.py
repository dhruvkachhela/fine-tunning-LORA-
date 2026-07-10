def ask(question, schema, max_new_tokens=128):
    prompt = f"""Given the database schema:
{schema}

Write a SQL query to answer: {question}"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

schema = "Table: employees (id, name, department, salary)"
question = "What is the average salary of employees in the Engineering department?"

print(ask(question, schema))

from datasets import load_dataset

dataset = load_dataset("xlangai/spider")
print(dataset)
print(dataset["train"][0])

schema_dataset = load_dataset("richardr1126/spider-schema")
print(schema_dataset)
print(schema_dataset["train"][0])

def build_prompt(question, schema_text):
    return f"""Given the database schema:
{schema_text}

Write a SQL query to answer: {question}

Respond with only the SQL query. No explanation, no markdown formatting."""



