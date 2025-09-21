#!/bin/bash
set -e

echo "--- Starting STS fine-tuned LADAM augmentation process ---"
echo "Step 0: Running LADAM augmentation"
python3 preprocess/ladams/ladam_sts.py
echo "--- Starting STS fine-tuned LADAM augmentation process ---"
echo "Step 1: Applying Faker-based PII substitution"
python3 preprocess/use_faker.py --input_file integrate_dataset_ladam.json
echo "Step 2: Correcting Korean particles"
python3 preprocess/process_sub.py --input_file integrate_dataset_ladam_faker.json

echo "Step 3-1: Converting to BIO format"
python3 preprocess/convert_bio.py --input_file integrate_dataset_ladam_faker_processed.json
echo "Step 4-1: Training bertmodel"
python3 train.py

echo "Step 3-2: Converting to BIO format"
python3 preprocess/convert_bio.py --input_file integrate_dataset_ladam_faker_processed.json --model 'klue/roberta-small'
echo "Step 4-2: Training roberta-small model"
python3 train_kor.py --model 'klue/roberta-small'

echo "Step 3-3: Converting to BIO format"
python3 preprocess/convert_bio.py --input_file integrate_dataset_ladam_faker_processed.json --model 'klue/roberta-base'
echo "Step 4-3: Training roberta-base model"
python3 train_kor.py --model 'klue/roberta-base'

echo "Step 3-4: Converting to BIO format"
python3 preprocess/convert_bio.py --input_file integrate_dataset_ladam_faker_processed.json --model 'klue/roberta-large'
echo "Step 4-4: Training roberta-large model"
python3 train_kor.py --model 'klue/roberta-large'

echo "--- Finished STS fine-tuned LADAM augmentation process ---"
