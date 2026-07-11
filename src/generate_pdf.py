import sys
import os
import subprocess

# Ensure reportlab is installed
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.pdfgen import canvas
except ImportError:
    print("[Info] reportlab is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page counts in the footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        
        # Draw running header & footer on all pages after the first
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Oblique", 8)
            self.setFillColor(colors.HexColor("#4A5568"))
            self.drawString(54, 750, "Technical Report: LoRA Fine-Tuning Qwen2.5-0.5B-Instruct for Text-to-SQL")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)
            
            # Footer
            self.setFont("Helvetica", 9)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawString(54, 36, "ML Portfolio Project")
            
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 54, 36, page_text)
            self.line(54, 48, 612 - 54, 48)
            
        self.restoreState()

def create_code_block(code_text, style):
    """
    Helper to wrap code in a formatted single-cell table with a grey background and borders.
    """
    escaped_code = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    p = Paragraph(f"<font face='Courier' size='8'>{escaped_code}</font>", style)
    t = Table([[p]], colWidths=[612 - 108])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    return t

def main():
    pdf_filename = "TECHNICAL_REPORT.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=1, # Centered
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'PaperSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-BoldOblique',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2B6CB0"),
        alignment=1,
        spaceAfter=15
    )
    
    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )
    
    body_bold = ParagraphStyle(
        'PaperBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    body_italic = ParagraphStyle(
        'PaperBodyItalic',
        parent=body_style,
        fontName='Helvetica-Oblique'
    )

    abstract_title_style = ParagraphStyle(
        'AbstractTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
        spaceAfter=5
    )
    
    abstract_text_style = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=4, # Justified
    )
    
    code_text_style = ParagraphStyle(
        'CodeText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1A202C")
    )
    
    story = []

    # --- COVER / TITLE HEADER ---
    story.append(Spacer(1, 20))
    story.append(Paragraph("Technical Report: Parameter-Efficient Fine-Tuning of Small Language Models for Text-to-SQL", title_style))
    story.append(Paragraph("A Low-Rank Adaptation (LoRA) Approach on the Spider Benchmark", subtitle_style))
    story.append(Paragraph("<b>Author:</b> ML Engineering & Portfolio Project &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> July 2026", author_style))
    story.append(Spacer(1, 10))

    # --- ABSTRACT BLOCK ---
    abstract_p = Paragraph(
        "<b>Abstract</b>—This report documents the design, implementation, and analysis of a parameter-efficient fine-tuning (PEFT) "
        "pipeline using Low-Rank Adaptation (LoRA) to adapt <i>Qwen2.5-0.5B-Instruct</i> for Natural Language to SQL (NL2SQL) translation. "
        "The model was trained on the cross-domain <b>Spider</b> benchmark. By training only <b>0.2184%</b> (1.08 million) of the model's "
        "parameters, we enforce structured SQL generation constrained to specific database schemas. Evaluated on 200 unseen databases "
        "from the Spider dev set, the model achieved a <b>21.5% Exact-Match (EM) accuracy</b>. Qualitative analysis reveals join-hallucination "
        "bias and logic reversal as primary failure modes, pointing to clear paths for future architecture scaling.",
        abstract_text_style
    )
    
    abstract_table = Table([[abstract_p]], colWidths=[612 - 148]) # Intentionally narrower
    abstract_table.setStyle(TableStyle([
        ('LINELEFT', (0,0), (-1,-1), 1.5, colors.HexColor("#2B6CB0")),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    
    story.append(abstract_table)
    story.append(Spacer(1, 20))

    # --- 1. INTRODUCTION ---
    story.append(Paragraph("1. Introduction & Motivation", h1_style))
    story.append(Paragraph(
        "Translating natural language questions into executable database queries (Text-to-SQL) is a cornerstone of intelligent database interaction. "
        "While large foundation models (>7B parameters) can perform zero-shot Text-to-SQL tasks, deploying them in edge or local environments is "
        "computationally expensive. This project investigates the capability of a highly compact parameter-efficient architecture—specifically, a "
        "500-million parameter model (<i>Qwen2.5-0.5B-Instruct</i>)—fine-tuned via LoRA.<br/><br/>"
        "The primary research objective is to teach the model <b>schema-linking</b> (aligning natural language entities with database table/column structures) "
        "and strict output formatting (SQL-only syntax, with zero conversational filler).",
        body_style
    ))

    # --- 2. DATASET ---
    story.append(Paragraph("2. Dataset and Preprocessing", h1_style))
    story.append(Paragraph(
        "<b>2.1 The Spider Benchmark vs. WikiSQL</b><br/>"
        "Unlike WikiSQL, which uses single-table databases and overlapping train/test schemas, <b>Spider</b> requires cross-database generalization. "
        "In Spider, the databases in the validation set have no overlap with those in the training set. To generalize, the model must read the schema "
        "dynamically, identify primary-foreign key relationships, and link them to the question.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 Data Pipeline Implementation</b><br/>"
        "Since Spider's base release does not contain raw text representations of schemas, we integrate a helper schema dataset "
        "<i>richardr1126/spider-schema</i>. The preprocessing pipeline is implemented in <i>src/fine_tune.py</i> through the following functions:<br/>"
        "1. <b>format_schema(db_id, schema_lookup)</b>: Extracts the list of tables, columns, and foreign key rules for a database.<br/>"
        "2. <b>build_prompt(question, schema_text)</b>: Combines the schema context, the user question, and formatting rules. It appends the instruction: "
        "<i>'Respond with only the SQL query. No explanation, no markdown formatting.'</i><br/>"
        "3. <b>format_example(row, schema_lookup)</b>: Structures the inputs into a standard multi-turn ChatML conversation schema.",
        body_style
    ))

    # --- 3. METHODOLOGY ---
    story.append(Paragraph("3. Low-Rank Adaptation (LoRA) Methodology", h1_style))
    story.append(Paragraph(
        "<b>3.1 Mathematical Principles</b><br/>"
        "Fine-tuning all 495 million weights of Qwen2.5-0.5B requires large GPU memory. LoRA solves this by freezing the pre-trained weights "
        "and representing the weight update as a low-rank factorization of two trainable matrices:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>&Delta;W = B &middot; A</b><br/>"
        "where <i>B</i> and <i>A</i> are low-rank matrices. The forward pass becomes: <i>h = W_0 x + (&alpha;/r) (B &middot; A) x</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3.2 Hyperparameter Configuration</b><br/>"
        "We applied the following Peft configuration to target the key attention matrices:<br/>"
        "&bull; <b>Rank (r):</b> 16 (controls the bottleneck dimension)<br/>"
        "&bull; <b>LoRA Alpha (&alpha;):</b> 32 (scaling coefficient)<br/>"
        "&bull; <b>Dropout:</b> 0.05 (preventing co-adaptation)<br/>"
        "&bull; <b>Target Modules:</b> <i>['q_proj', 'v_proj']</i> (restricting adaptation to Query and Value projections)<br/>"
        "&bull; <b>Trainable Parameters:</b> 1,081,344 / 495,114,112 (<b>0.2184%</b>)",
        body_style
    ))
    
    lora_code = """# LoRA Configuration used in PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj'],
    bias='none',
    task_type="CAUSAL_LM"
)"""
    story.append(create_code_block(lora_code, code_text_style))
    story.append(Spacer(1, 10))

    # --- 4. TRAINING PIPELINE ---
    story.append(Paragraph("4. Training Pipeline & Overfitting Analysis", h1_style))
    story.append(Paragraph(
        "<b>4.1 Configuration</b><br/>"
        "Training was performed using the Hugging Face <i>trl</i> library's <i>SFTTrainer</i>:<br/>"
        "&bull; <b>Dataset Size:</b> 7,000 training examples.<br/>"
        "&bull; <b>Effective Batch Size:</b> 8 (per-device batch size of 4, gradient accumulation steps of 2).<br/>"
        "&bull; <b>Learning Rate:</b> 2e-4.<br/>"
        "&bull; <b>Loss Masking:</b> SFT loss was evaluated strictly on assistant turn tokens, ignoring prompt schemas.",
        body_style
    ))
    story.append(Paragraph(
        "<b>4.2 Overfitting Identification</b><br/>"
        "Eval loss rose starting at step 100, while train loss kept falling (confirming overfitting). Training was stopped early and the step-100 weights were restored.",
        body_style
    ))

    # Training Loss Table
    table_data = [
        ["Step", "Training Loss", "Validation Loss"],
        ["100", "0.6205", "0.8678 (Best checkpoint)"],
        ["200", "0.4622", "0.9109"],
        ["300", "0.4064", "1.0753"],
        ["500", "0.2247", "1.2173"],
        ["800", "0.1771", "1.4191"]
    ]
    t = Table(table_data, colWidths=[80, 120, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F7FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EDF2F7")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # --- Page Break for Code Breakdown & Error Analysis ---
    story.append(PageBreak())

    # --- 5. CODEBASE WALKTHROUGH ---
    story.append(Paragraph("5. Detailed Code Walkthrough", h1_style))
    story.append(Paragraph(
        "<b>5.1 Training Logic (fine_tune.py)</b><br/>"
        "This script initializes the tokenizer, formats dataset rows to ChatML, wraps them using `apply_chat_template`, "
        "and sets up the trainer. It uses <i>dataset_text_field='text'</i> to run SFT on fully pre-formatted strings, "
        "saving checkpoints in <i>./nl2sql-lora-full</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>5.2 CLI Inference Utility (inference.py)</b><br/>"
        "Enables testing queries without loading full training modules. It loads the base causal LM and applies the "
        "LoRA adapter using PEFT's wrapper. It supports running predictions via direct CLI parameters.",
        body_style
    ))
    
    inf_code = """# Load base and attach LoRA adapter dynamically
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model, 
    "thefounder03/nl2sql-lora-qwen2.5-0.5b"
)"""
    story.append(create_code_block(inf_code, code_text_style))
    story.append(Spacer(1, 10))

    # --- 6. EVALUATION ---
    story.append(Paragraph("6. Evaluation Metrics & Results", h1_style))
    story.append(Paragraph(
        "On 200 held-out databases from the Spider dev set, the model achieved a <b>21.5% Exact-Match (EM) accuracy</b>. "
        "Exact-Match is extremely strict, checking character-by-character equivalence. It penalizes semantically valid SQL "
        "statements if they vary in capitalization, join ordering, or aliases from the ground truth.",
        body_style
    ))

    # --- 7. ERROR ANALYSIS ---
    story.append(Paragraph("7. Qualitative Error Analysis & Case Studies", h1_style))
    
    story.append(Paragraph("<b>7.1 Join-Hallucination Bias</b>", h2_style))
    story.append(Paragraph(
        "Because complex join queries dominate the training distribution, the model often adds unnecessary `JOIN` statements "
        "on prompts that only require a single table.",
        body_style
    ))
    
    join_example = """-- Question: Find the name of all singers.
-- Reference SQL:
SELECT name FROM singer;

-- Model Output (Hallucinated JOIN):
SELECT T1.name FROM singer AS T1 
JOIN concert AS T2 ON T1.singer_id = T2.singer_id;"""
    story.append(create_code_block(join_example, code_text_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>7.2 Reversed Logic/Constraint Errors</b>", h2_style))
    story.append(Paragraph(
        "The small base capacity (0.5B parameters) limits logical semantics, causing the model to swap sorting operators "
        "(e.g., ordering oldest-to-youngest using ASC instead of DESC).",
        body_style
    ))
    
    logic_example = """-- Question: List names and ages of singers from oldest to youngest.
-- Reference SQL:
SELECT name, age FROM singer ORDER BY age DESC;

-- Model Output (Reversed Logic):
SELECT name, age FROM singer ORDER BY age ASC;"""
    story.append(create_code_block(logic_example, code_text_style))
    story.append(Spacer(1, 10))

    # --- 8. DISCUSSION ---
    story.append(Paragraph("8. Discussion & Key Learnings", h1_style))
    story.append(Paragraph(
        "LoRA fine-tuning effectively handles structured output constraints (forcing the model to output SQL syntax only, "
        "bypassing chatty instructions) and improves schema entity linking. However, it does not solve core reasoning failures "
        "or eliminate structural hallucinations. To overcome these limitations, future work will scale parameters to "
        "Qwen2.5-7B using QLoRA and switch evaluation to execution-based metrics.",
        body_style
    ))
    
    story.append(Spacer(1, 10))

    # Build the document
    print("[Info] Compiling PDF...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print("[Success] PDF created successfully.")

if __name__ == "__main__":
    main()
