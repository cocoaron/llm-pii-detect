import json
import os
import re
from typing import List, Dict

label_types = [
    '{FULL_NAME}',
    '{EMAIL}',
    '{PHONE_NUM}',
    '{PASSPORT_NUM}',
    '{ADDRESS}',
    '{SOCIAL_SECURITY_NUM}',
    '{HEALTHCARE_NUM}',
    '{CARD_NUM}',
    '{SHIPMENT_TRACKING_NUM}',
    '{DRIVERS_LICENSE_NUM}'
]

def load_json(file_path):
    """Loads the JSON file and returns its content."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []  # Initialize with an empty structure if file is empty or invalid
    return []

def save_json(file_path, data):
    """Saves the updated JSON data back to the file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def tokenize_and_tag(sentence: str) -> List[Dict[str, str]]:
    """Tokenizes the sentence and applies BIO tagging for PII entities."""
    pii_tags = {
        'FULL_NAME': 'B-NAME',
        'EMAIL': 'B-EMAIL',
        'PHONE_NUM': 'B-PHONE',
        'PASSPORT_NUM': 'B-PASSPORT',
        'ADDRESS': 'B-ADDRESS',
        'SOCIAL_SECURITY_NUM': 'B-RRN',
        'HEALTHCARE_NUM': 'B-HEALTH',
        'CARD_NUM': 'B-CREDIT_CARD',
        'SHIPMENT_TRACKING_NUM': 'B-TRACKING',
        'DRIVERS_LICENSE_NUM': 'B-DRIVERS_LICENSE'
    }
    
    tokenized = []
    matches = re.finditer(r'\{(.*?)\}', sentence)
    last_end = 0
    
    for match in matches:
        before = sentence[last_end:match.start()].strip()
        if before:
            tokenized.extend(re.findall(r'\S+', before))
        
        pii_key = match.group(1)
        if pii_key in pii_tags:
            tokenized.append(f'{{{pii_key}}}')
        
        last_end = match.end()
    
    after = sentence[last_end:].strip()
    if after:
        tokenized.extend(re.findall(r'\S+', after))
    
    # Additional tokenization for parentheses
    final_tokens = []
    for token in tokenized:
        if re.search(r'[()]', token):
            parts = re.split(r'(\(|\))', token)
            final_tokens.extend([p for p in parts if p])
        else:
            final_tokens.append(token)
    
    labels = []
    for word in final_tokens:
        label = "O"  # Default label
        if re.match(r'\{(.*?)\}', word):
            pii_key = word.strip('{}')
            if pii_key in pii_tags:
                label = pii_tags[pii_key]
        labels.append({"word": word, "label": label})
    
    return labels

def add_sentence(file_path, sentence):
    """Appends a new sentence entry as a new list inside the JSON file, with tokenized BIO tagging."""
    data = load_json(file_path)
    
    new_entry = [
        {
            "sentence": sentence,
            "labels": tokenize_and_tag(sentence)
        }
    ]
    
    data.append(new_entry)  # Append as a new list
    save_json(file_path, data)
    print(f"Sentence added: {sentence}")

def replace_placeholders(sentence: str) -> str:
    for i in range(1, 11):
        sentence = sentence.replace(f'{{X{i}}}', label_types[i-1])
    return sentence

def process_sentences(json_file_path: str, new_sentence: str = None, txt_path: str = None):
    if new_sentence:
        sentence = replace_placeholders(new_sentence.strip())
        if sentence:
            add_sentence(json_file_path, sentence)
    elif txt_path:
        if not os.path.exists(txt_path):
            print(f"File not found: {txt_path}")
            return
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = replace_placeholders(line.strip())
                if sentence:
                    add_sentence(json_file_path, sentence)
    else:
        print("No input provided (new_sentence or txt_path required).")

# Example Usage
#json_file_path = "dataset/0314_dataset.json"  # Path to your JSON file
#json_file_path = "dataset/integrate_dataset_0402.json"

json_file_path = "dataset/new_integrate_dateset.json"

################################################

new_sentence = ""

################################################

txt_file_path = "./output/augmented_output.txt"#"0501_templates_new.txt"
process_sentences(json_file_path, txt_path=txt_file_path)




















"""

import json
import os

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []  # Initialize with an empty structure if file is empty or invalid
    return []

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def add_sentence(file_path, sentence):
\    data = load_json(file_path)
    
    new_entry = [
        {
            "sentences": sentence,
            "labels": []  # Empty label list for now
        }
    ]
    
    data.append(new_entry)  # Append as a new list
    save_json(file_path, data)
    print(f"Sentence added: {sentence}")


# Example Usage
json_file_path = "dataset/dataset.json"  # Path to your JSON file

new_sentence = "기후 변화는 환경에 심각한 영향을 미치고 있으며, {FULL_NAME}(이)가 거주하는 지역에서도 그 영향을 실감하고 있다. 최근 {EMAIL}로 수신된 환경 보고서에 따르면, 지구 온난화로 인해 해수면 상승이 가속화되고 있으며, 이는 해안 지역 주민들에게 직접적인 위협이 되고 있다(Lee et al., 2021). 특히, {PHONE_NUM} 번호를 사용하는 {FULL_NAME}은(는) 지난달 {SHIPMENT_TRACKING_NUM} 운송장을 통해 배송받은 기후 연구 자료에서 대기 중 이산화탄소 농도가 급격히 증가하고 있다는 사실을 확인했다(Smith & Jones, 2020).\
\
또한, 기후 변화로 인해 강수 패턴이 변화하면서 농업 생산성이 감소하고 있으며, 이는 식량 안보 문제로 이어지고 있다. {SOCIAL_SECURITY_NUM}를 보유한 {FULL_NAME}은(는) 농업 연구 기관과 협력하여 이를 해결하기 위한 방안을 모색 중이다(Kim, 2022). 더욱이, 최근 해외 출장을 다녀온 {FULL_NAME}은(는) 여권번호 {PASSPORT_NUM}를 사용하여 입국 심사를 통과했으며, 출장 중 기후 변화로 인해 발생한 대규모 산불의 피해를 목격했다(Brown, 2019).\
\
이러한 문제 해결을 위해 각국 정부와 기업들은 탄소 배출 저감 정책을 추진하고 있다. 하지만, {CARD_NUM}로 결제한 기후 변화 대응 세미나에서 논의된 바와 같이, 기존 산업 구조를 바꾸는 것은 쉽지 않은 과제이다(Johnson, 2023). 기후 변화의 영향은 점점 심각해지고 있으며, {FULL_NAME}을(를) 비롯한 많은 사람이 지속 가능한 미래를 위해 노력해야 할 필요성이 더욱 커지고 있다."

add_sentence(json_file_path, new_sentence)

"""