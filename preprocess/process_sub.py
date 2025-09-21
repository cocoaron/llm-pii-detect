import argparse
import json
import re
import copy

# 조사 판별 함수
def correct_particle(name, particle_type):
    has_jongseong = lambda char: (ord(char) - 44032) % 28 != 0
    if not name:
        return particle_type[1]
    last_char = name[-1]
    if particle_type == ("은", "는"):
        return "은" if has_jongseong(last_char) else "는"
    elif particle_type == ("이", "가"):
        return "이" if has_jongseong(last_char) else "가"
    elif particle_type == ("을", "를"):
        return "을" if has_jongseong(last_char) else "를"
    return particle_type[1]

particle_pairs = {
    "은": ("은", "는"),
    "는": ("은", "는"),
    "이": ("이", "가"),
    "가": ("이", "가"),
    "을": ("을", "를"),
    "를": ("을", "를"),
}

def clean_and_correct(dataset):
    result = copy.deepcopy(dataset)

    for entry in result:
        for item in entry:
            labels = item["labels"]
            sentence = item["sentence"]

            new_labels = []
            i = 0

            while i < len(labels):
                current = labels[i]
                # 조사 패턴: NAME, '(', 조사1, ')', 조사2
                if current["label"].startswith("B-NAME") and i + 4 < len(labels):
                    name = current["word"]
                    l1, l2, l3, l4 = labels[i+1:i+5]
                    if (
                        l1["word"] == "(" and
                        l2["word"] in particle_pairs and
                        l3["word"] == ")" and
                        l4["word"] in particle_pairs[l2["word"]]
                    ):
                        #print(f"\nFound: {name} {l1['word']} {l2['word']} {l3['word']} {l4['word']}\n")
                        
                        correct = correct_particle(name, particle_pairs[l2["word"]])
                        # 다양한 문장 내 패턴 고려: 띄어쓰기 또는 붙어 있는 경우
                        patterns = [
                            f"{name} ( {l2['word']} ) {l4['word']}",
                            f"{name}( {l2['word']} ) {l4['word']}",
                            f"{name} {l1['word']} {l2['word']} {l3['word']} {l4['word']}",
                            f"{name} {l1['word']} {l2['word']} {l3['word']} {l4['word']}",
                            f"{name}{l2['word']}{l4['word']}"  # fallback
                        ]
                        replaced = False
                        for pat in patterns:
                            if pat in sentence:
                                sentence = sentence.replace(pat, f"{name}{correct}", 1)
                                replaced = True
                                break
                        if not replaced:
                            # 마지막 수단: 괄호 제거 후 조사 중복 패턴이 직접적으로 들어간 경우
                            sentence = re.sub(
                                re.escape(name) + r"\([^\)]+\)" + re.escape(l4["word"]),
                                f"{name}{correct}",
                                sentence,
                                count=1
                            )

                        new_labels.append({"word": name, "label": current["label"]})
                        new_labels.append({"word": correct, "label": "O"})
                        i += 5
                        continue
                    elif (
                        l1["word"] in particle_pairs and
                        l2["word"] == "(" and
                        l3["word"] in particle_pairs[l1["word"]] and
                        l4["word"] == ")" 
                    ):
                        
                        #print(f"\nFound: {name} {l1['word']} {l2['word']} {l3['word']} {l4['word']}\n")
                        
                        correct = correct_particle(name, particle_pairs[l1["word"]])
                        # 다양한 문장 내 패턴 고려: 띄어쓰기 또는 붙어 있는 경우
                        patterns = [
                            f"{name} {l2['word']} ( {l3['word']} )",
                            f"{name}{l2['word']} ( {l3['word']} )",
                            f"{name}{l1['word']} {l2['word']} {l3['word']} {l4['word']}",
                            f"{name} {l1['word']} {l2['word']} {l3['word']} {l4['word']}",
                            f"{name}{l1['word']}{l3['word']}"  # fallback
                        ]
                        replaced = False
                        for pat in patterns:
                            if pat in sentence:
                                sentence = sentence.replace(pat, f"{name}{correct}", 1)
                                replaced = True
                                break
                        if not replaced:
                            # 마지막 수단: 괄호 제거 후 조사 중복 패턴이 직접적으로 들어간 경우
                            sentence = re.sub(
                                re.escape(name) + r"\([^\)]+\)" + re.escape(l4["word"]),
                                f"{name}{correct}",
                                sentence,
                                count=1
                            )

                        new_labels.append({"word": name, "label": current["label"]})
                        new_labels.append({"word": correct, "label": "O"})
                        i += 5
                        continue

                if current["word"] in ["(", ")"]:
                    i += 1
                    continue

                new_labels.append(current)
                i += 1

            item["labels"] = new_labels
            item["sentence"] = sentence

    return result



def process_json_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    cleaned = clean_and_correct(data)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(cleaned, outfile, ensure_ascii=False, indent=2)
        
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="input JSON file name")
args = parser.parse_args()

input_file = f"./dataset/{args.input_file}"  
output_file = f"{input_file[:-5]}_processed.json"

print("Preprocess started")
process_json_file(input_file, output_file)
print("Preprocess finished")