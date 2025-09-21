# -*- coding: utf-8 -*-
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# tokenizer 준비
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

# 입력 파일 경로
input_path = "./dataset/integrate_dataset_ladam_faker_processed.json"
output_path = "./dataset/converted_integrate_dataset_ladam_faker_processed.json"

# 라벨 맵 생성
label_map = {'O': 0}
label_id = 1

print("BIO label conversion w/ BERTtokenizerfast started")

# 라벨 수집
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for entry_list in data:
    for entry in entry_list:
        for label in entry['labels']:
            tag = label['label']
            tag_type = tag[2:] if tag.startswith(("B-", "I-")) else None
            if tag_type and f"B-{tag_type}" not in label_map:
                label_map[f"B-{tag_type}"] = label_id
                label_id += 1
                label_map[f"I-{tag_type}"] = label_id
                label_id += 1

print("Label map:", label_map)

def convert_entry(entry, label_map):
    sentence = entry["sentence"]
    word_labels = entry["labels"]  # [{word: ..., label: ...}]

    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=512,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    aligned_labels = []
    pii_spans = []

    # 각 PII 단어의 위치 (start, end) 수집
    for wl in word_labels:
        word = wl["word"]
        label = wl["label"]
        if label == "O":
            continue

        # 동일한 단어가 여러 번 등장할 수 있으므로 전체 매칭
        start_index = 0
        while True:
            idx = sentence.find(word, start_index)
            if idx == -1:
                break
            pii_spans.append({
                "start": idx,
                "end": idx + len(word),
                "label": label
            })
            start_index = idx + len(word)

    # token 별 BIO 라벨 부여
    for i, (start, end) in enumerate(offsets):
        if start == end:  # 특수 토큰 등
            aligned_labels.append(-100)
            continue

        matched = False
        for span in pii_spans:
            span_start = span["start"]
            span_end = span["end"]
            tag = span["label"]

            # 겹치기만 하면 매칭
            if end > span_start and start < span_end:
                tag_type = tag[2:]
                if start == span_start:
                    aligned_labels.append(label_map.get(f"B-{tag_type}", 0))
                else:
                    aligned_labels.append(label_map.get(f"I-{tag_type}", 0))
                matched = True
                break

        if not matched:
            aligned_labels.append(label_map["O"])

    return {
        "sentence": sentence,
        "labels": aligned_labels
    }



# 변환 실행
converted_data = []
for entry_list in tqdm(data):
    converted_batch = [convert_entry(entry, label_map) for entry in entry_list]
    converted_data.append(converted_batch)

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print("BIO label conversion w/ BERTtokenizerfast finished")