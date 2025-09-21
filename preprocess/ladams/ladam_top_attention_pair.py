# -*- coding: utf-8 -*-
import argparse
import torch
import random
import json
import math
import numpy as np
from transformers import AutoModel, AutoTokenizer
from konlpy.tag import Komoran
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('./model/klue_roberta-large/sts')
model = AutoModel.from_pretrained('./model/klue_roberta-large/sts', output_attentions=True).to(device)

model.eval()

file_path = "./dataset/integrate_dataset.json"

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
'''
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
        
    attentions = outputs.attentions
    # 첫 번째 레이어의 attention scores를 가져옴
    attention_scores = attentions[0][0]
    attention_scores = attention_scores.detach().cpu().numpy()

    src_len = len(source_words)
    tgt_len = len(target_words)
    
    # 원본 문장과 타겟 문장에 대한 attention score 계산
    cross_attention = attention_scores[:, :src_len, src_len:src_len + tgt_len].mean(axis=0)  

    # 토큰 갯수 확인
    src_token_lens = [len(tokenizer.tokenize(w)) for w in source_words]
    num_tokens = sum(src_token_lens)
    
    # 치환 토큰 개수 설정 (ladam 논문 기준)
    n_aug = math.ceil(num_tokens * 0.17)
    #print(f"# of Token: {num_tokens}, Swapped: {n_aug}")
    
    valid_pairs = []
    for i in range(src_len):
        if source_labels[i] != "O":
            continue
        for j in range(tgt_len):
            if target_labels[j] != "O":
                continue
            score = cross_attention[i][j]
            valid_pairs.append((i, j, score))

    if not valid_pairs:
        return None

    # 내림차순 attention score 기준 top-n_aug 쌍 선택
    top_pairs = sorted(valid_pairs, key=lambda x: -x[2])[:n_aug]

    used_src = set()
    new_labels = source_item["labels"].copy()

    for src_idx, tgt_idx, attn_score in top_pairs:
        if src_idx in used_src:
            continue  # 한 source 토큰에 대해서는 한 번만 치환
        replacement_token = target_words[tgt_idx]
        original_token = source_words[src_idx]

        if original_token not in source_item["sentence"]:
            continue  # 원문에 존재하지 않으면 skip

        new_labels[src_idx] = {
            "word": replacement_token,
            "label": source_labels[src_idx]
        }
        used_src.add(src_idx)

        if debug:
            print(f"[{src_idx}] {original_token} → {replacement_token} | score: {attn_score:.4f}")
    
    new_sentence = " ".join([label["word"] for label in new_labels])
    return {
        "sentence": new_sentence,
        "labels": new_labels
    }
'''

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
    
    '''
    encoding = tokenizer(
        source_sentence, target_sentence,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    '''
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
    '''
    valid_pairs = []
    for i in range(src_len):
        # get the index of the source word that matches the i of the source token and get the corresponding label
        if source_labels[some_index] != "O":
            continue
        for j in range(tgt_len):
            # get the index of the target word that matches the j of the target token and get the corresponding label
            if target_labels[some_index] != "O":
                continue
            score = cross_attention[i][j]
            valid_pairs.append((i, j, score))
    '''
    valid_pairs = []
    for i in range(src_len):
        src_word_idx = word_ids_src[i]
        if src_word_idx is None or source_labels[src_word_idx] != "O":
            continue

        for j in range(tgt_len):
            tgt_word_idx = word_ids_tgt[j]
            if tgt_word_idx is None or target_labels[tgt_word_idx] != "O":
                continue

            score = cross_attention[i][j]
            valid_pairs.append((i, j, score))
        
    if not valid_pairs:
        return None
    top_pairs = sorted(valid_pairs, key=lambda x: -x[2])[:n_aug]
    
    if debug:
        print(f"Top {len(top_pairs)} pairs based on attention scores:")
        for src_idx, tgt_idx, attn_score in top_pairs:
            src_word_idx = word_ids_src[src_idx]
            tgt_word_idx = word_ids_tgt[tgt_idx]

            src_word = source_words[src_word_idx] if src_word_idx is not None else "[None]"
            tgt_word = target_words[tgt_word_idx] if tgt_word_idx is not None else "[None]"

            print(f"Source: (word: {src_word}, word idx: {src_word_idx})(token: {src_tokens[src_idx]}, token idx: {src_idx})  | "
                  f"Target: (word: {tgt_word}, word idx: {tgt_word_idx})(token: {tgt_tokens[tgt_idx]}, token idx: {tgt_idx}) | "
                  f"Score: {attn_score:.4f}")     
        #exit(0)
    '''
    used_src = set()
    new_labels = source_item["labels"].copy()
    
    for src_idx, tgt_idx, attn_score in top_pairs:
        if src_idx in used_src:
            continue # 한 source 토큰에 대해서는 한 번만 치환
        src_tokens[src_idx] = tgt_tokens[tgt_idx]
        
        # recover word from tokens that are replaced

        # get the index of the source word that matches the src_idx of the src token
        
        new_labels[some_index] = {
            "word": recovered_word[some_index], # 치환된 단어
            "label": 'O'
        }
        used_src.add(src_idx)
    
    recovered_words = source_words.copy()

    for src_idx, tgt_idx, attn_score in top_pairs:

        src_word_idx = word_ids_src[src_idx]
        tgt_word_idx = word_ids_tgt[tgt_idx]
        
        # 더 낮은 attention score를 가지는 토큰으로 치환되지 않도록 체크
        if src_word_idx in used_src:
            continue
        if src_word_idx is None or tgt_word_idx is None:
            continue

        # 단어 단위로 치환
        # 토큰끼리 변환한 다음에 단어로 복원해야함
        # 
        
        new_labels[src_word_idx] = {
            "word": recovered_words[src_word_idx],
            "label": "O"
        }
        used_src.add(src_word_idx)

    new_sentence = " ".join([label["word"] for label in new_labels])
    
    return {
        "sentence": new_sentence,
        "labels": new_labels
    }
    '''
    # 토큰 치환
    modified_tokens = src_tokens.copy()
    used_src = set()

    for src_idx, tgt_idx, attn_score in top_pairs:
        src_word_idx = word_ids_src[src_idx]
        tgt_word_idx = word_ids_tgt[tgt_idx]
        if src_idx in used_src or src_word_idx is None or tgt_word_idx is None:
            continue
        modified_tokens[src_idx] = tgt_tokens[tgt_idx]
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

