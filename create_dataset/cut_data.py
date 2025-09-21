import json
import argparse

def split_by_max_len(labels, max_len):
    """
    단어 단위로 max_len 기준으로 문장을 분할
    """
    chunks = []
    current_chunk = []
    current_len = 0

    for pair in labels:
        word = pair['word']
        label = pair['label']
        word_len = len(word)

        if current_len + word_len + (1 if current_len > 0 else 0) > max_len:
            if current_chunk:
                sentence = ' '.join([p['word'] for p in current_chunk])
                chunks.append({
                    "sentence": sentence,
                    "labels": current_chunk
                })
            current_chunk = []
            current_len = 0

        current_chunk.append({"word": word, "label": label})
        current_len += word_len + (1 if current_len > 0 else 0)

    if current_chunk:
        sentence = ' '.join([p['word'] for p in current_chunk])
        chunks.append({
            "sentence": sentence,
            "labels": current_chunk
        })

    return chunks

def preprocess_json(input_path, output_path, max_len):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    for batch in raw_data:
        new_batch = []
        for item in batch:
            labels = item['labels']
            if len(labels) == 0:
                continue  # 비어있는 항목은 건너뜀
            if len(' '.join([p['word'] for p in labels])) <= max_len:
                # sentence 재구성하여 저장
                new_batch.append({
                    "sentence": ' '.join([p['word'] for p in labels]),
                    "labels": labels
                })
            else:
                new_batch.extend(split_by_max_len(labels, max_len))
        processed_data.append(new_batch)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Preprocessed data saved to: {output_path}")

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--max_len', type=int, default=150, help='Maximum sentence length (in characters)')
    args = parser.parse_args()

    preprocess_json(args.input, args.output, args.max_len)
'''
input = 'dataset/integrate_dataset.json'
max_len = 256
output = f'{input[:-5]}_{max_len}.json'

preprocess_json(input, output, max_len)