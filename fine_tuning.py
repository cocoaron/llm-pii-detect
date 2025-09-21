# -*- coding: utf-8 -*-
import re
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import unicodedata
from konlpy.tag import Komoran
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
komoran = Komoran()


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text) 
    text = re.sub(r'[\x00-\x1F\x7F\uFFFD]', '', text) 
    return text.strip()

def tokenize_dataset(sentence):
    no_placeholder_sentence = re.sub(r'\{[A-Z_]+\}', '', sentence)
    komoran = Komoran()
    pos_tags = komoran.pos(no_placeholder_sentence)
    tokens = []
    for word, tag in pos_tags:
        if tag in []:
        #['JX', 'JKB', 'JKS', 'XSA', 'XSN', 'XSV', 'ETM', 'JKO', 'EC', 
        #           'SP', 'MM', 'SS', 'SF', 'VV', 'VCP', 'VX', 'VCP', 'EF', 'EP', 'SE', 'ETN', 'JKG']:
            continue
        if " " in word:
            sub_tokens = word.split(" ")
            for sub_token in sub_tokens:
                if sub_token:
                    tokens.append(sub_token)
        else:
            tokens.append(word)
    return tokens

def get_encodings_custom(sentences1, sentences2, tokenizer, max_len=300):
    all_input_ids = []
    all_attention_masks = []

    vocab = tokenizer.get_vocab()
    pad_id = tokenizer.pad_token_id
    unk_id = tokenizer.unk_token_id

    for s1, s2 in tqdm(zip(sentences1, sentences2), total=len(sentences1), desc="Tokenizing"):
        source_tokens = tokenize_dataset(s1)
        target_tokens = tokenize_dataset(s2)
        all_tokens = source_tokens + target_tokens

        input_ids = [vocab.get(word, unk_id) for word in all_tokens][:max_len]
        attention_mask = [1] * len(input_ids)

        # �е�
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)

    return {
        "input_ids": torch.tensor(all_input_ids),
        "attention_mask": torch.tensor(all_attention_masks),
    }

class PairDataset(Dataset):
    def __init__(self, s1, s2, labels, tokenizer, task):
        self.encodings = tokenizer(s1,s2,padding=True,truncation=True,max_length=256,return_tensors="pt")
        #self.encodings = get_encodings_custom(s1, s2, tokenizer)
        self.labels = torch.tensor(labels, dtype=torch.float if task == "sts" else torch.long)
        self.task = task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def fine_tune_model(model_id, task="sts"):
    
    assert task in ["sts", "nli"]
    
    config = {
        "sts": {"batch_size": 16, "lr": 2e-5, "warmup": 214, "steps": 3596, "num_labels": 1},
        "nli": {"batch_size": 32, "lr": 1e-5, "warmup": 7318, "steps": 121979, "num_labels": 3}
    }[task]
 
    ds = load_dataset("kor_nlu", "sts" if task == "sts" else "nli", trust_remote_code=True)["train"]
    #s1 = ds["sentence1"] if task == "sts" else ds["premise"]
    #s2 = ds["sentence2"] if task == "sts" else ds["hypothesis"]
    #labels = [x / 5.0 for x in ds["score"]] if task == "sts" else ds["label"]
    
    s1_raw = ds["sentence1"] if task == "sts" else ds["premise"]
    s2_raw = ds["sentence2"] if task == "sts" else ds["hypothesis"]
    labels_raw = [x / 5.0 for x in ds["score"]] if task == "sts" else ds["label"]

    # �̻� ���� ���� �� None ����
    s1, s2, labels = zip(*[
        (clean_text(a), clean_text(b), l)
        for a, b, l in zip(s1_raw, s2_raw, labels_raw)
        if a is not None and b is not None
    ])
    s1, s2, labels = list(s1), list(s2), list(labels)

    print(f"\n====== {model_id} | Task: {task.upper()} ======")
   
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=config["num_labels"]).to(DEVICE)

    train_ds = PairDataset(s1, s2, labels, tokenizer, task)
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config["lr"], eps=1e-6, betas=(0.9, 0.98))
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config["warmup"],
        num_training_steps=config["steps"]
    )

    model.train()
    pbar = tqdm(range(config["steps"]))
    step = 0
    while step < config["steps"]:
        for batch in train_dl:
            if step >= config["steps"]: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_description(f"Step {step} | Loss: {loss.item():.4f}")
            pbar.update(1)
            step += 1

    model.eval()
    preds, trues = [], []
    eval_ds_raw = load_dataset("kor_nlu", "sts" if task == "sts" else "nli", trust_remote_code=True)["validation"]
    
    #s1_eval = eval_ds_raw["sentence1"] if task == "sts" else eval_ds_raw["premise"]
    #s2_eval = eval_ds_raw["sentence2"] if task == "sts" else eval_ds_raw["hypothesis"]
    #labels_eval = [x / 5.0 for x in eval_ds_raw["score"]] if task == "sts" else eval_ds_raw["label"]
    
    s1_raw = eval_ds_raw["sentence1"] if task == "sts" else eval_ds_raw["premise"]
    s2_raw = eval_ds_raw["sentence2"] if task == "sts" else eval_ds_raw["hypothesis"]
    labels_raw = [x / 5.0 for x in eval_ds_raw["score"]] if task == "sts" else eval_ds_raw["label"]

    s1_eval, s2_eval, labels_eval = zip(*[
        (clean_text(a), clean_text(b), l)
        for a, b, l in zip(s1_raw, s2_raw, labels_raw)
        if a is not None and b is not None
    ])
    s1_eval, s2_eval, labels_eval = list(s1_eval), list(s2_eval), list(labels_eval)

    eval_ds = PairDataset(s1_eval, s2_eval, labels_eval, tokenizer, task)
    eval_dl = DataLoader(eval_ds, batch_size=config["batch_size"])    
    for batch in eval_dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        if task == "sts":
            pred = output.logits.squeeze().cpu().numpy()
            preds.extend(pred)
            trues.extend(batch["labels"].cpu().numpy())
        else:
            pred = output.logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred)
            trues.extend(batch["labels"].cpu().numpy())

    if task == "sts":
        print(f"Spearman: {np.corrcoef(preds, trues)[0, 1]:.4f}")
    else:
        print(f"Accuracy: {accuracy_score(trues, preds):.4f}")

    # Save fine-tuned models for further analysis
    save_dir = os.path.join("model", model_id.replace("/", "_"), task)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved the model: {save_dir}")

models = [
    ("klue_roberta_large", "klue/roberta-large"),
]

if __name__ == "__main__":
    for _, model_id in models:
        fine_tune_model(model_id, task="sts")
        #fine_tune_model(model_id, task="nli")
