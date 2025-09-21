# LLM-PII-Detect  

This repository provides a framework to **automatically build, augment, and evaluate unstructured PII detection datasets** using LLM-based template synthesis and augmentation pipelines. The goal of this project is to create a scalable and flexible dataset pipeline for **Korean PII (Personally Identifiable Information) detection**, leveraging LLMs (e.g., ChatGPT) to synthesize PII placeholders, augment data with semantic variations, and evaluate on transformer-based models.  

---

## Components  

### 1. Prompt Generation for PII Templates (`/create_dataset/*')  
- Generate natural language sentences with **PII placeholders** (e.g., `{FULL_NAME}`, `{EMAIL}`).  
- Support diverse domains and contexts (finance, healthcare, government, etc.).  
- Export generated templates in structured JSON format.  

### 2. Dataset Construction (`/preprocess/*`)  
- Convert generated sentences into dataset structure with placeholders.
- **LADAM-based augmentation**: Layer-wise attention-driven augmentation.  
- **Faker-based injection**: Replace placeholders with realistic synthetic PII values.  
- **BIO Tagging**: Apply KLUE/Roberta tokenizer for token-level BIO labeling.  

### 4. Model Evaluation (`eval_transformers.sh')  
- Benchmark KLUE/Roberta series models on generated datasets.  
- Provide precision/recall/F1 evaluation scripts.  
- Enable direct comparison between augmentation strategies.  

---

## Current Design  
PII-bearing datasets are often scarce or privacy-restricted. This framework avoids handling real PII by generating synthetic datasets with placeholders and Faker-injected values.  

### Project pipeline
1. Generate unstructured sentences with placeholders using LLM prompts.  
2. Automatically construct dataset schema.  
3. Apply augmentation (LADAM, Faker, BIO tagging).  
4. Train and evaluate models (KLUE/Roberta).  

---

## Key Features  

- **LLM-based Template Synthesis** for diverse and realistic PII contexts.  
- **Augmentation Framework** combining LADAM and Faker injection.  
- **Evaluation Suite** for model comparison and reproducibility.  

