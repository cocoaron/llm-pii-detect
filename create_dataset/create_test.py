import random
import pandas as pd
from pathlib import Path
from datetime import datetime

def generate_random_prompts(n=10):
    label_types = [
        '{X1}',
        '{X2}',
        '{X3}',
        '{X4}',
        '{X5}',
        '{X6}',
        '{X7}',
        '{X8}',
        '{X9}',
        '{X10}'
    ]
    
    writing_styles = [
        'a critical analysis (with citations and references)',
        'an untitled blog (i.e., without a title) ',
        'a few paragraphs (without a title)',
        'informally and converstationally (as if you were speaking to a friend)',
        'formally and converstationally (as if you were in a service industry)',
        'formally (as if you were writing a classified information)',
        'unstructured (without any specific format)',
    ]
    # List of topics
    with open('./topics.txt') as f:
        topics = f.read()
    topics = topics.split('\n')

    df = pd.DataFrame()
    
    df['fields_used'] = [', '.join(random.sample(label_types, random.randint(1, 10))) for _ in range(n)]
    df['writing_style'] = [random.choice(writing_styles) for _ in range(n)]
    df['topic'] = [random.choice(topics) for _ in range(n)]
    df['majors'] = [random.choice(majors) for _ in range(n)]
    
    def get_few_shot_examples(selected_pii, pii_example_sentences, num_full_examples=2):
        all_examples = []

        for _ in range(num_full_examples):
            parts = []
            used = set()
            for p in selected_pii:
                if p in pii_example_sentences:
                    examples = pii_example_sentences[p]
                    if isinstance(examples, list):
                        ex = random.choice([e for e in examples if e not in used]) if len(examples) > 1 else examples[0]
                        parts.append(ex)
                        used.add(ex)
                    else:
                        parts.append(examples)
            if parts:
                input_line = ""#f"입력: 사용된 PII: {', '.join(selected_pii)}"
                output_line = f"{' '.join(parts)}"#f"출력: {' '.join(parts)}"
                all_examples.append(f"{input_line}\nExample #1: {output_line}")

        return "".join(all_examples)

    def create_prompt(data):
        
        placeholder_map = {
            '{X1}': '{X1} means \"full name placeholder\"',
            '{X2}': '{X2} means \"email placeholder\"',
            '{X3}': '{X3} means \"phone NUMBER placeholder\"',
            '{X4}': '{X4} means \"passport NUMBER placeholder\"',
            '{X5}': '{X5} means \"address placeholder\"',
            '{X6}': '{X6} means \"social security NUMBER placeholder\"',
            '{X7}': '{X7} means \"healthcare insurance NUMBER placeholder\"',
            '{X8}': '{X8} means \"card NUMBER placeholder\"',
            '{X9}': '{X9} means \"tracking NUMBER placeholder\"',
            '{X10}': '{X10} means \"driver\'s license NUMBER placeholder\"'
        }
        pii_example_sentences = {
            '{X1}': [  # 이름 (Full name)
            '{X1}이(가) 오늘 발표를 맡았습니다.',  # 이(가)
            '{X1}은(는) 어제 회의에 참석하지 않았습니다.',  # 은(는)
            '안녕하세요, {X1}님. 잠시 통화 가능하신가요?',  # 존댓말, 조사 없음
            '이번 일정은 {X1}이(가) 조율했습니다.',  # 이(가)
            '아, {X1} 왔구나! 여기 앉아.',  # 반말, 조사 없음
            '이건 {X1}은(는) 알만한 내용이잖아.'  # 은(는), 반말
            ],
            '{X2}': [  # 이메일
            '확인 메일은 {X2}으로(로) 발송해 드렸습니다.',  # 으로(로)
            '가입하신 이메일 주소는 {X2}이(가) 맞으신가요?',  # 이(가)
            '{X2}은(는) 더 이상 유효하지 않은 주소입니다.',  # 은(는)
            '내 메일은 {X2}야. 거기로 보내 줘.',  # 반말
            '{X2}로부터 회신이 오면 바로 처리해드리겠습니다.',  # 로부터 (파생 표현)
            '혹시 {X2}이(가) 맞는지 다시 한번 확인 부탁드립니다.'  # 이(가)
            ],
            '{X3}': [  # 전화번호
            '{X3}으로(로) 인증번호가 전송되었습니다.',  # 으로(로)
            '연락처는 {X3}이(가) 맞으신가요?',  # 이(가)
            '{X3}은(는) 현재 사용 중인 번호가 아닙니다.',  # 은(는)
            '야, 너 번호 아직도 {X3}이야?',  # 반말, 이(가)
            '나중에 {X3}으로(로) 전화 줄게.',  # 으로(로), 반말
            '상담원 연결은 {X3} 번호로 진행됩니다.'  # 조사 없음
            ],
            '{X4}': [  # 여권번호
            '등록된 여권번호는 {X4}입니다.',  # 존댓말, 조사 없음
            '여권 번호 {X4}이(가) 맞으신가요?',  # 이(가)
            '여권은 {X4} 번호로 발권 완료되었습니다.',  # 조사 없음 (명사 직접 수식)
            '예약할 때 여권번호를 {X4}으로(로) 등록했어?',  # 으로(로), 반말
            '다음 여권번호 {X4}은(는) 사용된 이력이 없습니다.'  # 은(는)
            ],
            '{X5}': [  # 주소
            '기본 배송지는 {X5}으로(로) 설정되어 있습니다.',  # 으로(로)
            '{X5}은(는) 현재 등록된 주소가 맞으신가요?',  # 은(는)
            '이사 간 주소가 {X5}이(가) 맞나요?',  # 이(가)
            '{X5}으로(로) 보내면 잘 도착할 거야.',  # 반말
            '서류는 다음 주소로 배송되었습니다: {X5}.',  # 조사 없음
            '내 예전 집 주소는 {X5}이었어.'  # 반말 + 이었어 (이/가 어형)
            ],
            '{X6}': [  # 주민등록번호
            '고객님의 주민등록번호 {X6}이(가) 확인되었습니다.',  # 이(가)
            '입력하신 주민등록번호 {X6}을(를) 검토 부탁드립니다.',  # 을(를)
            '주민등록번호 {X6}으로(로) 본인 확인이 가능합니다.',  # 으로(로)
            '{X6}이(가) 네 주민번호 맞아?',  # 반말
            '주민등록번호는 {X6}으로(로) 등록되어 있습니다.',  # 으로(로)
            '주민등록번호 {X6}은(는) 유효하지 않습니다.'  # 은(는)
            ],
            '{X7}': [  # 건강보험번호
            '건강보험번호 {X7}이(가) 등록되어 있습니다.',  # 이(가)
            '제출하신 번호 {X7}은(는) 유효합니다.',  # 은(는)
            '건강보험번호 {X7}으로(로) 청구가 진행 중입니다.',  # 으로(로)
            '{X7}은(는) 네 보험번호 맞지?',  # 반말
            '보험 기록은 {X7}으로(로) 조회할 수 있어.'  # 으로(로), 반말
            ],
            '{X8}': [  # 카드번호
            '결제는 카드번호 {X8}으로(로) 처리되었습니다.',  # 으로(로)
            '환불은 카드번호 {X8}으로(로) 진행될 예정입니다.',  # 으로(로)
            '카드번호 {X8}이(가) 네 거 맞아?',  # 반말
            '{X8}은(는) 만료된 카드입니다.',  # 은(는)
            '청구 내역은 {X8}을(를) 기준으로 확인됩니다.'  # 을(를)
            ],
            '{X9}': [  # 운송장번호
            '{X9}이(가) 맞는 운송장번호인지 확인 바랍니다.',  # 이(가)
            '운송장 번호 {X9}으로(로) 배송이 진행 중입니다.',  # 으로(로)
            '운송장 번호 {X9}이(가) 맞아?',  # 반말
            '운송 이력은 {X9}을(를) 통해 확인 가능합니다.',  # 을(를)
            '{X9}은(는) 어제 출고된 운송장 번호입니다.',  # 은(는)
            '택배는 {X9}으로(로) 발송되었습니다.'  # 조사 없음
            ],
            '{X10}': [  # 운전면허증번호
            '운전면허번호 {X10}이(가) 본인과 일치합니다.',  # 이(가)
            '면허 번호 {X10}은(는) 유효한 상태입니다.',  # 은(는)
            '{X10}으로(로) 운전자 자격 조회가 가능합니다.',  # 으로(로)
            '{X10}이(가) 네 면허번호 맞지?',  # 반말
            '운전 면허 번호 {X10}을(를) 입력해 주세요.',  # 을(를)
            '등록된 면허 번호는 {X10}이야. 확인해 봐.'  # 반말
            ]
        }
        
        selected_pii = data['fields_used'].split(', ')
        map = []
        selected_examples = []
        
        for i, pii in enumerate(selected_pii):
            if pii in placeholder_map:
                map.append(placeholder_map[pii])
            if pii in pii_example_sentences:
                #selected_examples.append(pii_examples[pii])
                selected_examples.append(pii_example_sentences[pii])
                
        data['fields_used'] = ', '.join(selected_pii)
        #data['examples'] = ', '.join(selected_examples)
        data['map'] = ', '.join(map)
        
        few_shot_examples = get_few_shot_examples(selected_pii, pii_example_sentences)

        # 프롬프트 생성
        prompt = (f"Be a dataset creation expert. Do not create generalized dataset. You are "
                  f"writing {data['writing_style']} on {data['topic']} with following PII data placeholders: {str(data['fields_used'])}. Each placeholder aligns with following PII tokens: {str(data['map'])}."
                  f"\nEnsure that the placeholder are naturally included throughout the text in the context as if they are corresponding real-life data. "
                    f"Treat placeholders as acutal PII data as they will be replaced by corresponding real-life data later."
                          f"\nWrite PII data in personal context. Do not ever generalize PII data. Do not include or hallucinate any other PII types that are not listed."
                              f"Write in Korean."
                                  f"\nDataset examples with PII placeholder examples, they only serve as superficial examples. Don't directly use the sentences in data creation: {few_shot_examples}")

        return prompt
        
    df['prompt'] = df.apply(lambda x: create_prompt(x), axis=1)
    
    return df

# Generate 10 random prompts
df_prompts = generate_random_prompts(100)

print(df_prompts)

Path('output').mkdir(exist_ok=True)
Path('output/csv').mkdir(exist_ok=True)
Path('output/prompts').mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'output/csv/prompts_{timestamp}.csv'
df_prompts.to_csv(filename, index=False)
print("Prompts saved to output/csv/prompts.csv")

filename = f'output/prompts/created_prompt.txt'
with open(filename, 'a') as f:  
    for prompt in df_prompts['prompt']:
        f.write(prompt + '\n\n')
