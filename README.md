# Qwen 2.5 0.5B Instruct Fine-Tuning Setup

This repository contains the setup files and scripts to load the **Qwen/Qwen2.5-0.5B-Instruct** model for fine-tuning.

## Repository Structure

- `requirements.txt`: Python package dependencies.
- `.gitignore`: Configured to ignore local caches, checkpoints, and model binaries.
- `src/load_model.py`: Script to load the model and tokenizer with GPU/CPU auto-mapping and float16 precision.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fine-tunning
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Load the Model:**
   To verify everything is installed correctly and download the model:
   ```bash
   python src/load_model.py
   ```
