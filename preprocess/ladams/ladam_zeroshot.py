# -*- coding: utf-8 -*-
import argparse
import torch
import random
import json
import math
import numpy as np
from transformers import AutoTokenizer, AutoModel

from konlpy.tag import Komoran
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
model = AutoModel.from_pretrained('klue/roberta-large', output_attentions=True).to(device)

model.eval()

file_path = "./dataset/integrate_dataset.json"
#file_path = "../../dataset/integrate_dataset.json"

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
    
    source_sentence = source_item["sentence"]
    target_sentence = target_item["sentence"]
    source_words = [l["word"] for l in source_item["labels"]]
    source_labels = [l["label"] for l in source_item["labels"]]
    target_words = [l["word"] for l in target_item["labels"]]
    target_labels = [l["label"] for l in target_item["labels"]]
   
    # tokenizer 적용 및 token 변환
    src_tokens = tokenizer.tokenize(source_sentence)
    tgt_tokens = tokenizer.tokenize(target_sentence)
    
    src_len = len(src_tokens)+2
    tgt_len = len(tgt_tokens)+1

    encoding = tokenizer(
        source_words,
        target_words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    input_ids = encoding['input_ids'][0]
    full_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    src_tokens = full_tokens[:src_len] # idx 꼬임 발생해서 재정의
    
    
    word_ids_all = encoding.word_ids()
    word_ids_src = word_ids_all[:src_len]
    word_ids_tgt = word_ids_all[src_len:src_len + tgt_len]


    outputs = model(**encoding)
        
    attentions = outputs.attentions
    # 첫 번째 레이어의 attention scores를 가져옴
    attention_scores = attentions[0][0]
    attention_scores = attention_scores.detach().cpu().numpy()
    
    # 원본 문장과 타겟 문장에 대한 attention score 계산
    cross_attention = attention_scores[:, :src_len, src_len:src_len + tgt_len].mean(axis=0)  
    
    # 토큰 갯수 확인
    # 치환 토큰 개수 설정 (ladam 논문 기준)
    n_aug = math.ceil((src_len-2) * 0.17) # [cls], [sep] 제외
    #print(f"# of Token: {num_tokens}, Swapped: {n_aug}")

    # 토큰 치환
    modified_tokens = src_tokens.copy()
    used_src = set()

    for _ in range(n_aug):
        src_idx = random.choice([i for i in range(src_len) if i not in used_src])
        tgt_idx = 0
        while True:
            src_idx = random.choice([i for i in range(src_len) if i not in used_src])
            src_word_idx = word_ids_src[src_idx]
            if src_word_idx is not None and source_labels[src_word_idx] == "O":
                break
        max = -1
        for j in range(tgt_len):
            tgt_word_idx = word_ids_tgt[j]
            if tgt_word_idx is not None and target_labels[tgt_word_idx] == "O":
                if max < cross_attention[src_idx][j]:
                    max = cross_attention[src_idx][j]
                    tgt_idx = j
                    
        modified_tokens[src_idx] = tgt_tokens[tgt_idx]
        if debug:
            attn_score = cross_attention[src_idx][tgt_idx]
            print(f"Source: (token: {src_tokens[src_idx]}, token idx: {src_idx})  | "
                  f"Target: (token: {tgt_tokens[tgt_idx]}, token idx: {tgt_idx}) | "
                  f"Score: {attn_score:.4f}")   
        # 치환한 토큰 인덱스 저장
        used_src.add(src_idx)

    # 토큰 치환 이후 단어 복원
    recovered_words = []
    current_word = ''
    prev_word_idx = None

    for token, word_idx in zip(modified_tokens, word_ids_src):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            if current_word:
                recovered_words.append(current_word)
            current_word = token.replace('##', '')
        else:
            current_word += token.replace('##', '')
        prev_word_idx = word_idx
    if current_word:
        recovered_words.append(current_word)

    # 라벨 업데이트
    new_labels = source_item["labels"].copy()
    for i in range(len(recovered_words)):
        new_labels[i] = {
            "word": recovered_words[i],
            "label": source_labels[i]
        }

    new_sentence = " ".join([label["word"] for label in new_labels])   
    if debug:
        print(f"Original: {source_sentence}")
        print(f"Modified: {new_sentence}")
        #print("Labels:")
        #for label in new_labels:
        #    print(f"{label['word']}: {label['label']}")
        exit(0)
    return {
        "sentence": new_sentence,
        "labels": new_labels
    }

augmented = []
n_aug = 5
example_printed = False

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=str, default=False, help="debug mode")
args = parser.parse_args()

debug = args.debug

for _ in range(n_aug):
    for i, s1 in enumerate(dataset):
        s2 = random.choice([s for j, s in enumerate(dataset) if j != i])
        if debug:
            aug = augment_item(s1, s2, debug=debug)
        else:
            aug = augment_item(s1, s2)
        if aug:
            augmented.append(aug)
            if debug:
                example_printed = True

with open(f"{file_path[:-5]}_ladam.json", "w", encoding="utf-8") as f:
    json.dump(dataset + augmented, f, ensure_ascii=False, indent=2)
    print("saved")

