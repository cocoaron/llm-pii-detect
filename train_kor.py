import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from tqdm import trange, tqdm


json_file_path = './dataset/converted_integrate_dataset_ladam_faker_processed.json'
data_name = json_file_path.split('.')[0]

# JSON 파일 읽기
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터 확인
print(f"First item: {data[0]}")

# 텍스트와 레이블 추출
texts = [sentence_data['sentence'] for batch in data for sentence_data in batch]
labels = [sentence_data['labels'] for batch in data for sentence_data in batch]

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label_sequence = self.labels[item]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 이미 정렬된 label 사용 (길이 보정 포함)
        if len(label_sequence) > self.max_len:
            aligned_labels = label_sequence[:self.max_len]
        else:
            aligned_labels = label_sequence + [-100] * (self.max_len - len(label_sequence))

        aligned_labels = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': aligned_labels
        }

class DWB_Loss(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super(DWB_Loss, self).__init__()
        self.class_weights = class_weights  # shape: [num_classes]

    def forward(self, logits, target):
        """
        logits: [batch_size, seq_len, num_classes]
        target: [batch_size, seq_len] with -100 for ignore_index
        """
        # [B, L, C] -> [B*L, C], target: [B*L]
        B, L, C = logits.shape
        logits = logits.view(-1, C)
        target = target.view(-1)

        # 마스크: ignore_index (-100) 제외
        mask = target != -100
        logits = logits[mask]       # [N, C]
        target = target[mask]       # [N]

        probs = F.softmax(logits, dim=-1)  # [N, C]
        log_probs = torch.log(probs + 1e-12)  # 안정성 확보

        # Gather 정답 클래스 확률: [N]
        p_t = probs[torch.arange(len(target)), target]
        log_p_t = log_probs[torch.arange(len(target)), target]

        # 정답 클래스의 가중치 가져오기
        w_t = self.class_weights[target]  # [N]
        weight_term = w_t ** (1 - p_t)

        # Cross-Entropy + Brier-style 보정항
        loss = -weight_term * log_p_t - p_t * (1 - p_t)

        return loss.mean()



num_labels = 21  # 레이블 개수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='klue/roberta-large')
args = parser.parse_args()

# BERT 모델과 토크나이저
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=num_labels).to(device)


MAX_LEN = 512
BATCH_SIZE = 8#16 kroberta 실패
LR=3e-5

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=93
)
test_data = [
    {"sentence": text, "labels": label}
    for text, label in zip(test_texts, test_labels)
]

test_save_path = f"test_split_{data_name}.json"
with open(test_save_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"Test set saved to {test_save_path}")


# 데이터셋 생성
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer, MAX_LEN)

# 데이터로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

# 학습 및 평가 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 동적 클래스 가중치 계산 및 적용
from collections import Counter
import numpy as np

flat_labels = [l for seq in train_labels for l in seq if l != -100]
label_counts = Counter(flat_labels)
total = sum(label_counts.values())
weights = [total / label_counts.get(i, 1) for i in range(num_labels)]
weights = np.array(weights)
weights = weights / np.max(weights) * 10.0  # 최대 가중치를 10으로 정규화
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# 훈련 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)    # 3e-5

# 학습 함수
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    for batch in tqdm(dataloader, desc="Training batch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
        
        # nn.CrossEntropyLoss 사용
        # reshape: [B*L, C], labels: [B*L]
        B, L, C = logits.shape
        logits = logits.view(-1, C)
        labels = labels.view(-1)
        # loss 계산
        loss = loss_fn(logits, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


# 평가 함수
def evaluate(model, dataloader, device):
    model = model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
            predictions = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy().flatten())
            pred_labels.extend(predictions.cpu().numpy().flatten())

    # 각 레이블에 대한 precision, recall, f1-score 계산
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, labels=list(range(num_labels)),
                                                                     zero_division=0)

    # 레이블 0의 가중치를 낮추기 위해 가중치를 수동으로 설정
    custom_weights = support.copy()
    custom_weights[0] *= 0  # 레이블 0의 가중치를 낮춤

    # 가중 평균 계산
    weighted_precision = (precision * custom_weights).sum() / custom_weights.sum()
    weighted_recall = (recall * custom_weights).sum() / custom_weights.sum()
    weighted_f1 = (f1 * custom_weights).sum() / custom_weights.sum()

    return precision, recall, f1, support, weighted_precision, weighted_recall, weighted_f1

EPOCHS = 100#200
EARLY_STOPPING_THRESHOLD = 10#20
CHANGE_LR_EPOCH = 80#100
REDUCED_LR = 1e-5
'''
EPOCHS = 20
EARLY_STOPPING_THRESHOLD = 5
CHANGE_LR_EPOCH = 10
REDUCED_LR = 1e-5
'''

best_f1 = 0
best_accuracy = 0
early_stopping_count = 0
reduced_lr_applied = False

for epoch in trange(EPOCHS, desc='Epoch'):
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
    precision, recall, f1, support, weighted_precision, weighted_recall, weighted_f1 = evaluate(model, test_dataloader, device)
    
    print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {train_loss}, Weighted F1: {weighted_f1}')

    if epoch >= CHANGE_LR_EPOCH:
        if weighted_f1 < best_f1:
            early_stopping_count += 1
        else:
            early_stopping_count = 0

        if not reduced_lr_applied and early_stopping_count > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = REDUCED_LR
            reduced_lr_applied = True
            print(f"Learning rate reduced to {REDUCED_LR} at epoch {epoch + 1}")

        if early_stopping_count >= EARLY_STOPPING_THRESHOLD:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    best_f1 = max(best_f1, weighted_f1)

# 평가
precision, recall, f1, support, weighted_precision, weighted_recall, weighted_f1 = evaluate(model, test_dataloader, device)

# 각 레이블에 대한 결과 출력
for label in range(num_labels):
    print(
        f"Label {label} - Precision: {precision[label]}, Recall: {recall[label]}, F1-Score: {f1[label]}, Support: {support[label]}")

# 가중 평균 결과 출력
print(f"Weighted Precision: {weighted_precision}, Weighted Recall: {weighted_recall}, Weighted F1-Score: {weighted_f1}")


data_name = json_file_path.split('.')[0]

model_save_path = f"bert_with_{data_name}_hyper_seq{MAX_LEN}_epoch{EPOCHS}_lr{LR}_batch{BATCH_SIZE}.pth"


torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")