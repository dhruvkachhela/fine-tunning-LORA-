# Technical Report: Parameter-Efficient Fine-Tuning of Small Language Models for Text-to-SQL
**A Low-Rank Adaptation (LoRA) Approach on the Spider Benchmark**

---

## Abstract
This report documents the design, implementation, and analysis of a parameter-efficient fine-tuning (PEFT) pipeline using Low-Rank Adaptation (LoRA) to adapt `Qwen2.5-0.5B-Instruct` for Natural Language to SQL (NL2SQL) translation. The model was trained on the cross-domain **Spider** benchmark. By training only **0.2184%** (1.08 million) of the model's parameters, we enforce structured SQL generation constrained to specific database schemas. Evaluated on 200 unseen databases from the Spider dev set, the model achieved a **21.5% Exact-Match (EM) accuracy**. Qualitative analysis reveals join-hallucination bias and logic reversal as primary failure modes, pointing to clear paths for future architecture scaling.

---

## 1. Introduction & Motivation
Translating natural language questions into executable database queries (Text-to-SQL) is a cornerstone of intelligent database interaction. While large foundation models (>7B parameters) can perform zero-shot Text-to-SQL tasks, deploying them is computationally expensive. This project investigates the capability of a highly compact parameter-efficient architecture—specifically, a 500-million parameter model (`Qwen2.5-0.5B-Instruct`)—fine-tuned via LoRA.

The primary research objective is to teach the model **schema-linking** (aligning natural language entities with database table/column structures) and strict output formatting (SQL-only syntax, with zero conversational filler).

---

## 2. Dataset and Preprocessing

### 2.1 The Spider Benchmark vs. WikiSQL
Unlike WikiSQL, which uses single-table databases and overlapping train/test schemas, **Spider** requires cross-database generalization. In Spider, the databases in the validation set have no overlap with those in the training set. To generalize, the model must read the schema dynamically, identify primary-foreign key relationships, and link them to the question.

### 2.2 Data Pipeline Implementation
Since Spider's base release does not contain raw text representations of schemas, we integrate a helper schema dataset `richardr1126/spider-schema`. The preprocessing pipeline is implemented in the Jupyter notebook [fine-tune-kaggle.ipynb](file:///c:/Users/dhruv/Downloads/GROWN%20WINGS/fine-tunning/src/fine-tune-kaggle.ipynb) through the following functions:

1. **`format_schema(db_id, schema_lookup)`**: Extracts the list of tables, columns, and foreign key rules for a database.
   * *Example Output:* `Schema: employee: id (number), name (text), dept_id (number) | department: id (number), name (text) ... Foreign Keys: employee: dept_id equals department: id`
2. **`build_prompt(question, schema_text)`**: Combines the schema context, the user question, and formatting rules. It appends the instruction: *"Respond with only the SQL query. No explanation, no markdown formatting."*
3. **`format_example(row, schema_lookup)`**: Structures the inputs into a standard multi-turn conversation format (ChatML schema):
   ```json
   {
     "messages": [
       {"role": "user", "content": "<schema_info>\nWrite a SQL query to answer: <question>\nRespond with only the SQL query..."},
       {"role": "assistant", "content": "<sql_query>"}
     ]
   }
   ```
4. **`apply_template(example)`**: Tokenizes the messages and applies Qwen's ChatML template, producing a raw string wrapper complete with start and end system tokens (`<|im_start|>` and `<|im_end|>`).

---

## 3. Low-Rank Adaptation (LoRA) Methodology

### 3.1 Mathematical Principles
Fine-tuning all 495 million weights of Qwen2.5-0.5B requires large GPU memory and runs the risk of catastrophic forgetting. LoRA solves this by freezing the pre-trained weights ($W_0 \in \mathbb{R}^{d \times k}$) and representing the weight update ($\Delta W$) as a low-rank factorization of two trainable matrices:

$$\Delta W = B \cdot A$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with the rank $r \ll \min(d, k)$. The forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

where $\alpha$ is a scaling hyperparameter that stabilizes optimization when altering $r$.

### 3.2 Hyperparameter Configuration
We applied the following Peft configuration to target the key attention matrices:
* **Rank ($r$):** 16 (controls the bottleneck dimension of $A$ and $B$)
* **LoRA Alpha ($\alpha$):** 32 (scaling coefficient)
* **Dropout:** 0.05 (preventing co-adaptation of LoRA weights)
* **Target Modules:** `['q_proj', 'v_proj']` (restricting adaptation to the Query and Value projection layers in the self-attention mechanism)
* **Trainable Parameters:** 1,081,344 / 495,114,112 (**0.2184%**)

---

## 4. Training Pipeline

### 4.1 Configuration
Training was performed via the Hugging Face `trl` library using `SFTTrainer` and `SFTConfig`:
* **Dataset Size:** 7,000 training examples.
* **Effective Batch Size:** 8 (per-device batch size of 4 with gradient accumulation steps set to 2).
* **Learning Rate:** $2 \cdot 10^{-4}$ (AdamW optimizer).
* **Precision:** `fp16` (Float16 precision for training stability on standard GPUs).
* **Loss Masking:** Supervised Fine-Tuning (SFT) was set up such that the loss function is only evaluated on the labels generated during the **assistant turn**. No gradient update is computed on the user schema prompt, preventing the model from diluting its language model capabilities on structured table names.

### 4.2 Overfitting Analysis and Early Stopping
The evaluation loss on a 200-sample validation subset was calculated every 100 steps. 

```
Loss
 │
2.00 ┼
     │
1.50 ┼                                      ● Validation Loss (1.4191 at step 800)
     │                                     /
1.00 ┼             ● Best Checkpoint (0.8678 at step 100)
     │            / \
0.50 ┼  ●───●────●───●───────●───────●────● Train Loss (0.1771 at step 800)
     │  
0.00 ┴──┴───┴────┴───┴───────┴───────┴────┴──
       100 200  300 400     500     700  800   Steps
```

Starting immediately after **Step 100**, the validation loss steadily climbed (from `0.8678` up to `1.4191` at step 800) while the training loss continued to plunge (reaching `0.1771`). This rapid divergence confirmed that the model was memorizing schema names and specific query shapes of the training set rather than learning generalized schema linking. 
To prevent severe overfitting, training was terminated early, and the weights from **Step 100** were restored via `load_best_model_at_end=True`.

---

## 5. Codebase Walkthrough

The codebase is organized as follows:

### 5.1 [fine-tune-kaggle.ipynb](file:///c:/Users/dhruv/Downloads/GROWN%20WINGS/fine-tunning/src/fine-tune-kaggle.ipynb) (The Jupyter Training Notebook)
* Contains the complete training logic, validation checks, and experimental code.
* Loads the Hugging Face dataset, formats database schemas, tokenizes inputs using ChatML templates, applies the LoRA configuration, and executes supervised fine-tuning (SFT) training logs.

### 5.2 [inference.py](file:///c:/Users/dhruv/Downloads/GROWN%20WINGS/fine-tunning/src/inference.py) (The CLI Inference Agent)
* Designed for runtime prediction.
* Accepts parameters for `--schema`, `--question`, `--base_model`, and `--adapter_id`.
* Employs PEFT's `PeftModel.from_pretrained` to load the base Qwen model and apply the weights downloaded from your Hugging Face adapter repository: `thefounder03/nl2sql-lora-qwen2.5-0.5b`.
* Automates prompt structuring and runs token-generation to output clean SQL.

---

## 6. Evaluation Metrics & Results
We evaluated the Step 100 restored model on 200 held-out dev questions:
* **Exact-Match (EM) Accuracy:** **21.5%** (43 / 200 queries matched the reference SQL character-for-character).

### Understanding Exact-Match (EM) Stricter Constraints
EM is a stringent metric compared to **Execution Accuracy (EX)**:
* EM checks if the predicted string matches the target string (often including syntax ordering).
* For example, if the reference is `SELECT name, age FROM person`, and the model predicts `SELECT age, name FROM person`, the EM is `0` (failure), even though both queries execute and produce the exact same database records.
* Thus, a 21.5% EM accuracy represents a solid foundation in learning output syntax and structural database constraints.

---

## 7. Qualitative Error Analysis & Case Studies

A deep audit of the validation failures highlighted two major patterns of cognitive errors.

### 7.1 Join-Hallucination Bias
* **Hypothesis:** Because the training set contains many complex multi-table SQL prompts, the fine-tuned model becomes biased toward linking tables. It assumes that if a schema specifies a foreign key relation, it must write a `JOIN` statement, even when the question can be resolved using a single table.
* **Case Study:**
  * *Question:* *"Find the name of all singers."*
  * *Schema:* `singer: singer_id, name, age | concert: concert_id, name, singer_id (Foreign Key)`
  * *Reference SQL:* `SELECT name FROM singer`
  * *Model Prediction:*
    ```sql
    SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id;
    ```
  * *Analysis:* The model accurately links the entities but constructs an unnecessary, expensive join with the `concert` table.

### 7.2 Reversed Logic/Constraint Errors
* **Hypothesis:** The small parameter footprint (0.5B) restricts the model's command of logical semantics, causing it to confuse ordering directions or comparison boundaries.
* **Case Study:**
  * *Question:* *"List names and ages of singers from oldest to youngest."*
  * *Reference SQL:* `SELECT name, age FROM singer ORDER BY age DESC`
  * *Model Prediction:*
    ```sql
    SELECT name, age FROM singer ORDER BY age ASC;
    ```
  * *Analysis:* The model maps the correct columns and applies sorting but outputs ascending order (`ASC`) instead of descending (`DESC`), reversing the user's intent.

---

## 8. Discussion & Key Learnings

| What LoRA Fine-Tuning Fixes | What LoRA Fine-Tuning Does NOT Fix |
| :--- | :--- |
| **Output Syntax Control**: The model stops producing conversational filler or markdown markers and generates raw SQL queries. | **Implicit Semantic Logic**: Weak logical mapping (such as reversing sorting directions or using incorrect inequalities). |
| **Basic Schema Matching**: Learning how to map table names and select columns according to schema definitions. | **Hallucinations of Relations**: The tendency to write joins that are not required by the user prompt. |

### 8.1 Future Directions
1. **Scale Base Model Parameter Capacity**: Moving from a `0.5B` to a `1.5B` or `7B` parameter base model will increase logical reasoning and contextual parsing capacity, lowering logical reversal rates.
2. **QLoRA Fine-Tuning**: Allow training on larger base models using 4-bit quantization, making it possible to run larger configurations on personal GPU accelerators.
3. **Execution Accuracy Metrics**: Integrate a database evaluator (such as `sqlite3` in python) to evaluate outputs based on query results rather than strict string comparisons.
