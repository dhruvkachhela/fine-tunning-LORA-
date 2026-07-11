# nl2sql-lora

This repository contains the code and configuration for a Parameter-Efficient Fine-Tuning (PEFT) project that fine-tunes **Qwen2.5-0.5B-Instruct** for natural-language-to-SQL (NL2SQL) generation using **LoRA (Low-Rank Adaptation)**.

The model is fine-tuned on the cross-domain **Spider** benchmark to translate user queries into valid SQL statements matching a specified database schema.

* **Hugging Face Model Adapter Repository:** [thefounder03/nl2sql-lora-qwen2.5-0.5b](https://huggingface.co/thefounder03/nl2sql-lora-qwen2.5-0.5b)
* **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct`

---

## Overview

This project applies Low-Rank Adaptation (LoRA) to fine-tune `Qwen2.5-0.5B-Instruct` on the **Spider** benchmark, a complex, cross-domain natural-language-to-SQL dataset. Spider comprises 200 databases spanning 138 distinct domains, requiring models to dynamically map natural language questions onto database schemas (including tables, columns, types, and primary/foreign keys) to generate valid, executable SQL queries.

---

## Why Spider over WikiSQL?

When developing natural-language-to-SQL models, dataset selection dictates whether the model learns real **schema-linking** or simply memorizes pattern mappings. 
* **WikiSQL** is a large dataset but is limited to simple, single-table queries. The databases in its train and test splits overlap heavily, making it too simple. Indeed, modern instruction-tuned LLMs can solve WikiSQL with high zero-shot accuracy.
* **Spider** presents a much harder generalization challenge: the databases used in the training, development, and test splits **never overlap**. This forces the model to generalize to entirely unseen database structures (cross-database generalization) and perform structural reasoning (e.g., locating foreign key relationships for multi-table joins), rather than memorizing schema identifiers.

---

## Approach

### Model & LoRA Parameters
We target the projection matrices in the attention blocks of the Qwen transformer layers using LoRA:
* **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct`
* **LoRA Rank ($r$):** 16
* **LoRA Alpha ($\alpha$):** 32
* **LoRA Dropout:** 0.05
* **Target Modules:** `['q_proj', 'v_proj']`
* **Trainable Parameters:** `1,081,344` out of `495,114,112` total parameters (**0.2184%**)

```python
# LoRA Configuration used in PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj'],
    bias='none',
    task_type="CAUSAL_LM"
)
```

### Dataset & Format
Because the base release of Spider does not bundle raw schema text with each row (only schema structures in SQLite databases), schema mappings were retrieved from `richardr1126/spider-schema`. They are formatted with table columns, types, and explicit foreign key relationships.

#### Prompt Template (ChatML)
Each training sample is formatted into standard ChatML user/assistant turns. To enforce zero-shot-like behavior and format consistency, an explicit directive is appended to the instruction:

```markdown
<|im_start|>user
Given the database schema:
Schema:
department : Department_ID (number) , Name (text) , Creation (text) | head : head_ID (number) , name (text) , age (number) | management : department_ID (number) , head_ID (number)
Foreign Keys:
management : head_ID equals head : head_ID | management : department_ID equals department : Department_ID

Write a SQL query to answer: How many heads of the departments are older than 56 ?

Respond with only the SQL query. No explanation, no markdown formatting.<|im_end|>
<|im_start|>assistant
SELECT count(*) FROM head WHERE age > 56<|im_end|>
```

The model is trained via `trl.SFTTrainer` with **label-masking**, meaning loss is computed strictly on the assistant's tokens (the target SQL query), ignoring the system prompts and user schemas.

---

## Baseline (Pre-Fine-Tuning) Behavior

Before fine-tuning, the base `Qwen2.5-0.5B-Instruct` model successfully solves simple, single-table tasks. However, it fails schema-linking and multi-table operations. For example, given a query requiring a join on employee departments:

* **Question:** *"Find the name and department of all employees and the budget of their departments."*
* **Baseline Output:**
  ```sql
  -- Syntactically valid but semantically broken grouping/join selection:
  SELECT name, T1.name, budget 
  FROM employees 
  JOIN department AS T1 ON employees.id = T1.id 
  GROUP BY employees.id;
  ```
  *The base model joined on `employees.id = department.id` (incorrect primary-foreign key linking) and grouped by `employees.id` instead of joining on `department_id` and grouping by `department_id`.*

---

## Training Run

* **Dataset Size:** 7,000 training examples
* **Hyperparameters:** Learning Rate `2e-4`, effective batch size `8` (batch size 4, gradient accumulation steps 2).
* **Early Stopping:** Training was monitored via evaluation loss on a 200-sample validation subset.
* **Overfitting Identification:** Eval loss began climbing starting at step 100, while training loss consistently fell. 

| Step | Training Loss | Validation Loss |
| :--- | :--- | :--- |
| **100** | **0.6205** | **0.8678** (Best checkpoint) |
| 200 | 0.4622 | 0.9109 |
| 300 | 0.4064 | 1.0753 |
| 500 | 0.2247 | 1.2173 |
| 800 | 0.1771 | 1.4191 |

Training was halted early. The best checkpoint (Step 100) was automatically restored and saved using `load_best_model_at_end=True`.

---

## Evaluation

We evaluated the restored step-100 model checkpoint on **200 held-out validation databases** from the Spider dataset.

* **Exact-Match Accuracy:** **21.5%** (43 / 200 examples)
* **Metrics Note:** *Exact-Match (EM)* is a highly strict metric. It compares the literal token sequences of the predicted SQL to the reference SQL. It penalizes semantically-identical queries (e.g. `SELECT name, age FROM human` vs `SELECT age, name FROM human`, or different spacing/capitalization in aliases) that would otherwise yield the exact same database results under *Execution Accuracy*.

---

## Error Analysis

A manual audit of the model's exact-match failures revealed two prominent structural and reasoning bugs:

### 1. Join-Hallucination Bias
The model frequently hallucinated table relationships and added redundant `JOIN` operations. If a schema contains relations, the model overgeneralizes from its training distribution (where complex multi-table questions dominate) and incorporates tables that are completely irrelevant to the prompt.
* **Question:** *"Find the name of all singers."*
* **Schema:** `singer : singer_ID (number), Name (text) | concert : concert_ID (number), name (text)`
* **Model Output:**
  ```sql
  SELECT T1.Name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_ID = T2.singer_ID;
  ```
  *The query should simply select from `singer`, but the model aggressively joined the `concert` table despite the question only asking for the singers' names.*

### 2. Reversed Logic Errors
The model occasionally exhibits logical inverse errors when compiling ordering or filtering directives. This represents a genuine constraint reasoning failure rather than a syntax format error.
* **Question:** *"Show the names and ages of the singers, ordered from oldest to youngest."*
* **Model Output:**
  ```sql
  SELECT Name, Age FROM singer ORDER BY Age ASC;
  ```
  *The question asks for oldest-to-youngest, which requires descending (`DESC`) order, but the model output `ASC`.*

---

## Limitations & Future Work

1. **Parameter Limits:** A `0.5B` parameter base model is extremely small for complex context retrieval, limits logical synthesis capacity, and struggles to balance instruction-following with massive schema tables.
2. **Short Training Duration:** Early stopping at step 100 capped adaptation progress.
3. **Evaluation Metric:** Shifting from strict exact-match comparison to actual **Execution Accuracy** using SQLite environments would reflect the model's true query capabilities.

### Future Experiments
* **QLoRA on Larger Base Models:** Scale up to `Qwen2.5-1.5B` or `Qwen2.5-7B` using 4-bit quantization (QLoRA) to improve semantic parsing capability.
* **Refined Hyperparameters:** Lower the learning rate (e.g., `5e-5`) and increase training epochs with cosine learning rate schedules to find a better validation loss minimum.
* **Execution Eval Suite:** Integrate SQLite execution-based test runners to run queries against database files.

---

## Key Learnings

| What LoRA Fine-Tuning Fixes | What LoRA Fine-Tuning Does NOT Fix |
| :--- | :--- |
| **Enforces Output constraints:** Standardizes outputs to clean SQL, eliminating verbose markdown/text explanations. | **Reasoning bugs:** Struggles with inverse sorting requests (`ASC` vs `DESC`) or complex logical constraints. |
| **Improves Schema Alignment:** Structural understanding of keys and table link syntax. | **Hallucinations:** Cannot prevent the model from assuming tables or keys must be joined when simple queries suffice. |
| **Adaptation Speed:** Achieved 21.5% accuracy using under 1.1 million parameters trained on single T4 GPUs. | **Absolute context limits:** Struggles to parse giant database schemas containing dozens of tables. |

---

## Repository Structure

```markdown
├── README.md                 # Project documentation and analysis
├── TECHNICAL_REPORT.md       # Deep-dive technical report (Markdown)
├── TECHNICAL_REPORT.pdf       # Compiled academic-style PDF report
├── requirements.txt          # Python package requirements
└── src/
    ├── fine-tune-kaggle.ipynb # Original notebook with Kaggle logs and metrics
    ├── generate_pdf.py       # PDF report generator script
    └── inference.py          # CLI inference script to run NL2SQL queries
```

---

## Getting Started

### 1. Installation
Ensure PyTorch is installed with CUDA support, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Inference
You can load the fine-tuned adapter directly from Hugging Face and run natural-language queries:
```bash
python src/inference.py \
  --schema "Table: student (id, first_name, last_name, age, major_id) | Table: major (id, name)" \
  --question "Find the names of students majoring in Computer Science."
```
