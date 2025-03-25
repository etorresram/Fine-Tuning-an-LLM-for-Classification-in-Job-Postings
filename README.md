# Fine-Tuning an LLM with an Additional Layer for Multi-Class Economic Sector Classification in Job Postings

This project applies a fine-tuned Large Language Model (LLM), specifically DistilBERT, to classify job vacancy descriptions into economic sectors based on NAICS codes. The goal is to map unstructured text data (job postings) to structured sector classifications using state-of-the-art natural language processing (NLP) techniques.

## Overview

- **Model**: DistilBERT (a lightweight version of BERT)
- **Task**: Multi-class classification (21 economic sectors)
- **Input**: Job description text (`description` column)
- **Output**: Predicted NAICS sector code (`naics_code`, mapped to labels 0–20)
- **Dataset Size**: ~5,000 job postings

## Key Features

- Fine-tuning a pretrained transformer (LLM) with a custom classification head
- PyTorch-based training and evaluation loops
- Support for GPU acceleration
- Includes preprocessing, tokenization, label encoding, and train/test splitting
- Compatible with both Google Colab and local Python environments (e.g., VS Code)

## Requirements

- Python 3.8+
- PyTorch
- Transformers (`transformers` by Hugging Face)
- scikit-learn
- pandas

Install dependencies:
```bash
pip install torch transformers scikit-learn pandas
```

## How It Works

1. Load a CSV dataset with job descriptions and NAICS codes
2. Encode NAICS codes into numerical labels (0–20)
3. Tokenize text using DistilBERT tokenizer
4. Fine-tune a `DistilBertForSequenceClassification` model
5. Train and evaluate on 80/20 train/test split
6. Output performance metrics (loss and accuracy)

## File Structure

- `DistilBERT_Sectors.py`: Main training script
- `jobs_sectors.csv`: Input dataset with columns `description` and `naics_code`
- `README.md`: Project overview and instructions

## Example Use

To train the model:
```bash
python DistilBERT_Sectors.py
```

The script will print training and validation accuracy/loss for each epoch.

## Acknowledgements

- Hugging Face Transformers
- NAICS (North American Industry Classification System)
- Open source NLP community



