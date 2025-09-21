# -*- coding: utf-8 -*-
import torch
import random
import json
import math
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

file_path = "./dataset/integrate_dataset.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('./model/klue_roberta-large/sts')
model = AutoModel.from_pretrained('./model/klue_roberta-large/sts', output_attentions=True).to(device)
model.eval()

print("LADAM started")

with open(file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = []
for group in raw_data:
    dataset.extend(group) 


def find_high_attention_O_token(sentence_words, attention_scores, labels):
    o_indices = [i for i, label in enumerate(labels) if label == "O"]
    if not o_indices:
        return None
    avg_scores = attention_scores.mean(axis=0) 
    best_idx = max(o_indices, key=lambda i: avg_scores[i])
    return best_idx

def augment_item(source_item, target_item, debug=False):
    source_words = [l["word"] for l in source_item["labels"]]
    source_labels = [l["label"] for l in source_item["labels"]]
    target_words = [l["word"] for l in target_item["labels"]]
    target_labels = [l["label"] for l in target_item["labels"]]

    encoding = tokenizer(
        source_words, target_words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
        
    with torch.no_grad():
        outputs = model(**encoding)

    attentions = outputs.attentions[0] 
    avg_attention = attentions[0].mean(dim=0).cpu().numpy()

    src_len = len(source_words)
    tgt_len = len(target_words)
    
    # 원본 문장과 타겟 문장에 대한 attention score 계산
    cross_attention = avg_attention[:src_len, src_len:src_len + tgt_len]

    # 토큰 갯수 확인
    src_token_lens = [len(tokenizer.tokenize(w)) for w in source_words]
    num_tokens = sum(src_token_lens)
    
    # 치환 토큰 개수 설정 (ladam 논문 기준)
    n_aug = math.ceil(num_tokens * 0.17)
    #print(f"# of Token: {num_tokens}, Swapped: {n_aug}")

    o_index = [i for i, label in enumerate(source_labels) if label == "O"]
    if not o_index:
        return None

    # O 라벨에 해당하는 원본 문장 토큰 인덱스를 랜덤하게 선택
    selected = random.sample(o_index, min(n_aug, len(o_index)))

    new_labels = source_item["labels"].copy()
    new_sentence = source_item["sentence"]

    # 랜덤하게 선택한 토큰 index에 대해 치환
    for src_idx in selected:
        
        # 타겟 문장에서 cross attention이 가장 높은 O 라벨을 가진 토큰 확인
        target_attn_weights = cross_attention[src_idx]
        tgt_o_indices = [i for i, label in enumerate(target_labels) if label == "O"]
        if not tgt_o_indices:
            continue
        tgt_best = max(tgt_o_indices, key=lambda i: target_attn_weights[i])

        # 치환
        replacement_token = target_words[tgt_best]
        original_token = source_words[src_idx]

        # 중복 치환 방지
        if original_token not in new_sentence:
            continue

        new_labels[src_idx] = {
            "word": replacement_token,
            "label": source_labels[src_idx]
        }
        new_sentence = " ".join([label["word"] for label in new_labels])

        if debug:
            print(f"[{src_idx}] {original_token} -> {replacement_token} | highest attention: {target_attn_weights[tgt_best]:.4f}")
    if debug:
        print(f"Original: {source_item['sentence']}")
        print(f"Labels: {source_item['labels']}")
        print(f"Augmented: {new_sentence}")
        print(f"Labels: {new_labels}")
    return {
        "sentence": new_sentence,
        "labels": new_labels
    }

augmented = []
n_aug = 1#5
example_printed = False

for _ in range(n_aug):
    for i, s1 in enumerate(dataset):
        s2 = random.choice([s for j, s in enumerate(dataset) if j != i])
        debug = not example_printed  # 첫 번째 예시에 대해서만 출력
        aug = augment_item(s1, s2)#, debug=debug)
        if aug:
            augmented.append(aug)
            if debug:
                example_printed = True

with open(f"{file_path[:-5]}_ladam.json", "w", encoding="utf-8") as f:
    json.dump(dataset + augmented, f, ensure_ascii=False, indent=2)
    print("LADAM saved")

