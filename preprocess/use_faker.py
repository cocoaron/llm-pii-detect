import argparse
import json
import string
from faker import Faker
from datetime import datetime, timedelta
import json
import random
import re

# Initialize Faker with Korean locale
fake = Faker('ko_KR')
fake_en = Faker('en_US')

class Email:
    def getContext(email):
        templates = [
            "{email}",
            "이메일 {email}",
            "이메일 주소 {email}",
            "이메일 정보 {email}",
            "이메일 주소 정보 {email}",
        ]
        #return random.choice([email, templates]).format(email=email)
        return email
    
    def generate_valid_email():
        """유효한 이메일 주소 생성 함수"""
        name_patterns = [
            lambda: ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(6, 12))),
            lambda: f"{random.choice(string.ascii_lowercase)}{random.randint(1000, 9999)}",
            lambda: f"{''.join(random.choices(string.ascii_lowercase, k=2))}{random.randint(10, 99)}",
            lambda: f"{''.join(random.choices(string.ascii_lowercase, k=3))}_{random.randint(100, 999)}",
            lambda: f"{''.join(random.choices(string.ascii_lowercase, k=3))}.{random.randint(10, 999)}",
            lambda: f"{''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 6)))}{random.choice(['x', '_', '.', ''])}{random.randint(1, 9999)}"
        ]
        example_domains = [
            "@aol.com", "@apple.com", "@att.net", "@amazon.com", "@adobe.com", "@archlinux.org", "@airmail.net", "@alumni.harvard.edu", "@alumni.stanford.edu", "@alum.mit.edu", "@ameritech.net", "@asia.com", "@bk.ru",
            "@bigpond.com", "@bell.net", "@blueyonder.co.uk", "@bellsouth.net", "@bolt.com", "@buffalo.edu", "@byu.edu", "@berkeley.edu", "@basilicmail.com", "@bluemail.ch", "@boltmail.com", "@chol.com", "@chollian.net",
            "@citizen.seoul.kr", "@cyworld.com", "@chungbuk.ac.kr", "@chungnam.ac.kr", "@cbu.ac.kr", "@chosun.ac.kr", "@cam.ac.uk", "@cityu.edu.hk", "@cantab.net", "@college.harvard.edu", "@daum.net", "@dreamwiz.com",
            "@dongguk.edu", "@dr.com", "@dmail.com", "@discroot.org", "@dartmouth.edu", "@dmu.ac.uk", "@docomo.ne.jp", "@dupont.com", "@disroot.org", "@devnullmail.com", "@empas.com", "@ewha.ac.kr", "@earthlink.net",
            "@europe.com", "@eircom.net", "@email.cz", "@estu.edu", "@eml.cc", "@ethz.ch", "@epost.de", "@exe.com", "@epub.com", "@freechal.com", "@fastmail.com", "@fun.com", "@freenet.de", "@foxmail.com", "@fau.edu",
            "@fmail.com", "@fau.org", "@forty4.com", "@freenetname.co.uk", "@fmail.ru", "@fantasymail.com", "@gmail.com", "@googlemail.com", "@gmx.com", "@gachon.ac.kr", "@gawab.com", "@gmx.us", "@gsu.edu", "@gatorzone.com",
            "@glover.com", "@glasgow.ac.uk", "@gbridge.org", "@hanafos.com", "@hanmail.net", "@hanmir.com", "@hanyang.ac.kr", "@hotmail.com", "@hushmail.com", "@hawaii.edu", "@howard.edu", "@hunter.cuny.edu",
            "@hqmail.com", "@husky.neu.edu", "@hinet.net", "@icloud.com", "@inbox.com", "@icmail.net", "@iu.edu", "@imperial.ac.uk", "@iit.edu", "@iemail.com", "@inbox.ru", "@istanbul.edu", "@ibm.net", "@ieee.org",
            "@iowa.gov", "@jejunu.ac.kr", "@juno.com", "@japan.com", "@japanmail.com", "@jredmail.com", "@juilliard.edu", "@jh.edu", "@jacobs-university.de", "@jawamail.com", "@jourrapide.com", "@javeriana.edu.co", "@juniv.edu",
            "@kakao.com", "@korea.kr", "@kw.ac.kr", "@korea.ac.kr", "@kookmin.ac.kr", "@kaist.ac.kr", "@kyonggi.ac.kr", "@ktu.edu", "@kent.edu", "@kth.se", "@keio.jp", "@kabelmail.de", "@live.com", "@lycos.co.kr", "@list.ru",
            "@laposte.net", "@lavabit.com", "@liverpool.ac.uk", "@lsu.edu", "@lu.se", "@luther.edu", "@love.com", "@london.com", "@latinmail.com", "@msn.com", "@mail.jason.pe.kr", "@mail.ru", "@mail.com", "@me.com",
            "@mailinator.com", "@mac.com", "@myway.com", "@monash.edu", "@mayo.edu", "@muni.cz", "@myself.com", "@narasarang.or.kr", "@nate.com", "@naver.com", "@netsgo.com", "@netian.com", "@nctu.edu.tw", "@nju.edu.cn",
            "@nu.edu", "@northwestern.edu", "@ncsu.edu", "@nyu.edu", "@netzero.net", "@outlook.com", "@orange.fr", "@ole.com", "@opera.com", "@offmail.com", "@openmailbox.org", "@o2.pl", "@oakland.edu", "@ohio-state.edu",
            "@okc.edu", "@ou.edu", "@ostmail.com", "@paran.com", "@protonmail.com", "@postech.ac.kr", "@pitt.edu", "@purdue.edu", "@polymail.io", "@pm.me", "@pmail.com", "@prodigy.net", "@peoplepc.com", "@pacbell.net",
            "@phmail.com", "@q.com", "@qq.com", "@qmail.com", "@qu.edu", "@queen.edu", "@qub.ac.uk", "@querymail.com" "@qmail.co.uk", "@qsl.net", "@qyale.edu", "@quik.com", "@qut.edu.au", "@rediffmail.com", "@rambler.ru",
            "@rocketmail.com", "@rochester.edu", "@rutgers.edu", "@rus.com", "@romail.com", "@redhat.com", "@runbox.com", "@ryerson.ca", "@rice.edu", "@rwu.edu", "@snu.ac.kr", "@startmail.com", "@seznam.cz", "@seoultech.ac.kr",
            "@sogang.ac.kr", "@skku.edu", "@skhu.ac.kr", "@swu.edu", "@stanford.edu", "@samsung.com", "@scu.edu", "@sbcglobal.net", "@skhu.ac.kr", "@tistory.com", "@tutanota.com", "@tutanota.de", "@texas.edu",
            "@temple.edu", "@tnu.ac.kr", "@tokyo.ac.jp", "@techie.com", "@tmail.com", "@tucows.com", "@tnstate.edu", "@tmail.org", "@uos.ac.kr", "@uow.ac.kr", "@ucla.edu", "@uchicago.edu", "@uiuc.edu", "@uva.edu",
            "@utoronto.ca", "@ubc.ca", "@umich.edu", "@uol.com", "@umass.edu", "@verizon.net", "@virgin.net", "@virginia.edu", "@vivaldi.net", "@vivamail.com", "@vt.edu", "@vu.edu", "@vrmail.net", "@vmail.co",
            "@vanderbilt.edu", "@vu.lt", "@vuoz.com", "@wanadoo.fr", "@windstream.net", "@walla.co.il", "@wisc.edu", "@wits.ac.za", "@warwick.ac.uk", "@web.de", "@worldemail.com", "@wmail.com", "@washington.edu",
            "@welcomemail.com", "@woalla.com", "@xmail.net", "@xtra.co.nz", "@xfinity.com", "@xobni.com", "@xemaps.com", "@xoxomail.com", "@xiamen.edu", "@xcu.edu", "@xoxo.org", "@xbridge.com", "@xen.org", "@xcyber.com",
            "@yahoo.com", "@yandex.com", "@yonsei.ac.kr", "@ymail.com", "@y7mail.com", "@yifan.net", "@yopmail.com", "@ygroupmail.com", "@yale.edu", "@yola.com", "@york.ac.uk", "@yieldmail.com", "@zoho.com", "@zmail.com",
            "@zmail.org", "@zju.edu.cn", "@zulipmail.com", "@ziggo.nl", "@zohomail.in", "@zmail.ru", "@zeelandnet.nl", "@zimbra.com", "@zwallet.com", "@zoznam.sk"
        ]
        common_kr_domains = [
            "@naver.com", "@daum.net", "@gmail.com", "@hanmail.net", "@nate.com", "@hotmail.com", "@yahoo.com",
            "@kakao.com"
        ]

        username = random.choice(name_patterns)()
        com_domain = random.choice(common_kr_domains)
        ex_domain = random.choice(example_domains)

        domain = random.choice([com_domain, ex_domain])

        email = f"{username}{domain}"#random.choice([f"{username}{domain}", fake.email()])
        # return Email.getContext(email)
        return email
    
class PhoneNumber: 
    def getContext(number):
        templates = [
            "번호 {number}",
            "전화번호 {number}",
            "연락처 {number}",
            "전화 {number}",
            "전화정보 {number}",
            "연락받으실 번호 {number}",
            "통화가능 번호 {number}",
            "연락 가능 번호 {number}",
            "연락정보 {number}"
            #"PHONE NUMBER {number}",
            #"Phone {number}",
            #"Contact Number {number}",
            #"Tel {number}",
        ]
        #return random.choice([number, templates]).format(number=number)
        return number
        
    # Function to generate a Korean phone number
    def generate_korean_phone_number():
        """한국 전화번호(휴대폰 및 유선전화) 생성 함수"""
        mobile_prefixes = ['010', '011', '016', '017', '018', '019']
        landline_prefixes = {
            '02': '서울', '031': '경기', '032': '인천', '033': '강원',
            '041': '충남', '042': '대전', '043': '충북', '044': '세종',
            '051': '부산', '052': '울산', '053': '대구', '054': '경북',
            '055': '경남', '061': '전남', '062': '광주', '063': '전북',
            '064': '제주'
        }

        if random.choice([True, False]):  # 50% 확률로 휴대폰 또는 유선전화 선택
            prefix = random.choice(mobile_prefixes)
            middle = f"{random.randint(0, 9999):04d}"
        else:
            prefix = random.choice(list(landline_prefixes.keys()))
            middle = f"{random.randint(0, 999):03d}" if prefix != '02' else f"{random.randint(0, 9999):04d}"

        last = f"{random.randint(0, 9999):04d}"
    
        #number = f"{prefix}-{middle}-{last}"
        format = [f"{prefix}-{middle}-{last}", f"{prefix}{middle}{last}", f"{prefix} {middle} {last}"]
        number = random.choice(format)
        #return PhoneNumber.getContext(number)
        return number

class Passport:
    def getContext(number):
        templates = [
            "여권번호 {number}",
            "여권 {number}",
            "여권 정보 {number}",
            "여권정보 {number}",
            "여권번호 {number}",
            "여권 번호 {number}",
            "여권번호 정보 {number}",
            "여권번호 정보 {number}",
            "여권번호 정보 {number}",
            "여권번호 정보 {number}"
            #"PASSPORT NUMBER {number}",
            #"Passport {number}",
            #"Passport Info {number}",
            #"Passport No {number}",
        ]
        #return random.choice([number, templates]).format(number=number)
        return number
    
    # Function to generate a Korean passport number
    def generate_korean_passport_number():
        """한국 여권번호를 생성하는 단일 함수"""
        number = random.choice([f"{random.choice(['M', 'R'])}{random.randint(0, 999999):06d}", fake.passport_number()])
        #return Passport.getContext(number)
        return number

class CreditCard:   
    def getContext(number):
        templates = [
            "카드번호 {number}",
            "신용카드 {number}",
            "카드 번호 {number}",
            "결제카드 번호 {number}",
            "카드정보 {number}",
            "결제수단 {number}",
            "등록된 카드 {number}",
            "결제카드 {number}",
            "카드 일련번호 {number}",
            "결제정보 {number}"
            
            #"CARD NUMBER {number}",
            #"Credit Card: {number}",
            #"Payment method: {number}",
            #"Card No: {number}",
        ]
        #return random.choice([number, templates]).format(number=number)
        return number
    
    # Function to generate a Korean credit card number
    def calculate_luhn_checksum(number: str) -> int:
        """Luhn 알고리즘을 사용하여 체크섬 계산"""
        digits = [int(d) for d in number]
        for i in range(len(digits) - 1, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        total = sum(digits)
        return (10 - (total % 10)) % 10
    
    def generate_korean_credit_card_number():
        """한국 신용카드 번호를 생성하는 단일 함수"""
        card_types = {
            'Visa': {'prefix': ['4'], 'length': 16},
            'MasterCard': {'prefix': ['51', '52', '53', '54', '55'], 'length': 16},
            'BC카드': {'prefix': ['94', '95'], 'length': 16},
            '삼성카드': {'prefix': ['94'], 'length': 16},
            '현대카드': {'prefix': ['95'], 'length': 16},
            '롯데카드': {'prefix': ['92'], 'length': 16},
            '신한카드': {'prefix': ['91'], 'length': 16},
            'KB국민': {'prefix': ['93'], 'length': 16}
        }

        card_type = random.choice(list(card_types.keys()))
        prefix = random.choice(card_types[card_type]['prefix'])
        length = card_types[card_type]['length']

        number = prefix + ''.join(str(random.randint(0, 9)) for _ in range(length - len(prefix) - 1))
        checksum = CreditCard.calculate_luhn_checksum(number)

        card_number = number + str(checksum)
        if random.choice([True, False]):
            card_number = '-'.join([card_number[i:i+4] for i in range(0, len(card_number), 4)])
        #return CreditCard.getContext(card_number)
        return card_number

class KoreanRRN:
    def getContext(number):
        templates = [
            "주민등록번호 {number}",
            "주민번호 {number}",
            "주민등록정보 {number}",
            "주민등록 번호 {number}",
            "{number} 번호",
            #"RRN {number}",
            #"RRN NUMBER {number}",
            #"Korean RRN {number}",
        ]
        #return random.choice([number, templates]).format(number=number)
        return number
    
    def generate_random_date():
        """1900년부터 현재까지 랜덤한 생년월일 생성"""
        start_date = datetime(1900, 1, 1)
        end_date = datetime.now()
        days_between_dates = (end_date - start_date).days
        random_date = start_date + timedelta(days=random.randint(0, days_between_dates))
        return random_date

    def get_gender_digit(year):
        """출생 연도에 따른 성별 코드 반환"""
        if 1900 <= year <= 1999:
            return random.choice([1, 2, 5, 6])
        else:
            return random.choice([3, 4, 7, 8])

    def calculate_checksum(rrn):
        """Luhn 알고리즘과 유사한 주민등록번호 체크섬 계산"""
        multipliers = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
        checksum = sum(int(rrn[i]) * multipliers[i] for i in range(12))
        return (11 - (checksum % 11)) % 10

    def generate_korean_rrn():
        """한국 주민등록번호를 생성하는 단일 함수"""
        # 주민등록번호 생성 로직
        birth_date = KoreanRRN.generate_random_date()
        gender_digit = KoreanRRN.get_gender_digit(birth_date.year)
        region_code = random.randint(0, 95)  # 지역 코드 (0~95)
        serial = random.randint(0, 999)  # 개인 일련번호 (000~999)
    
        rrn_base = f"{birth_date.strftime('%y%m%d')}{gender_digit}{region_code:02d}{serial:03d}"
        checksum = KoreanRRN.calculate_checksum(rrn_base)
    
        number = f"{rrn_base}{checksum}"
        if random.choice([True, False]):
            number = f"{number[:6]}-{number[6:]}"
        #return KoreanRRN.getContext(number)
        return number

class DriverLicense:

    def getContext(number):
        templates = [
            "운전면허번호 {number}",
            "면허번호 {number}",
            "운전면허 번호 {number}",
            "면허정보 {number}",
            "면허증번호 {number}",
            "운전면허증 번호 {number}",
            "{number} 번호",
            #"DRIVER LICENSE {number}",
            #"LICENSE {number}",
            #"Korean Driver License {number}",
            #"Driver's License Number {number}"
        ]
        #return random.choice([number, templates]).format(number=number)
        return number
    
    # Function to generate a Korean driver's license number
    def generate_korean_driver_license():
        """한국 운전면허번호를 생성하는 단일 함수"""
        region_codes = [str(i).zfill(2) for i in range(11, 29)]  # 11-28
        issue_years = [str(i).zfill(2) for i in range(0, 100)]  # 00-99

        region = random.choice(region_codes)
        issue_year = random.choice(issue_years)
        serial = f"{random.randint(0, 999999):06d}"
        check_digit = f"{random.randint(0, 99):02d}"

        number = random.choice([f"{region}{issue_year}{serial}{check_digit}", f"{region}-{issue_year}-{serial}-{check_digit}"])
        #return DriverLicense.getContext(number)
        return number

class Address:
    def getContext(address):
        templates = [
            "{address}"
        ]
        return random.choice(templates).format(address=address)
    
    def generate_korean_address():
        """한국 주소를 생성하는 단일 함수"""
        
        address = fake.address()
        
        if random.random() < 0.5:
            # 50% 확률로 '길'을 '번길'로 치환, 숫자+길/로/거리 -> 숫자+번길
            address = re.sub(r'(\d{1,3})길', r'\1번길', address)
            address = re.sub(r'(\d{1,3})로', r'\1번길', address)
            address = re.sub(r'(\d{1,3})거리', r'\1번길', address)
            
        return Address.getContext(address)
    
class ShipmentTracking:
    def getContext(number):
        templates = [
            "송장번호 {number}",
            "운송장번호 {number}",
            "송장번호 정보 {number}",
            "운송장 정보 {number}",
            "운송장번호 정보 {number}",
            "배송번호 {number}",
            "배송정보 {number}",
        ]
        #return random.choice([number, templates]).format(number=number)
        return number
    
    def generate_shipment_tracking_number():
        """운송장 번호 생성 함수"""
        #return ShipmentTracking.getContext(fake.ean13())
        return fake.ean13()

class HealthcareNumber:
    def getContext(number):
        templates = [
            "건강보험번호 {number}",
            "건강보험 {number}",
            "건강보험 정보 {number}",
            "건강보험정보 {number}",
            "건강보험번호 정보 {number}",
            "건강보험증번호 {number}",
            "건강보험증 번호 {number}",
        ]
        #return random.choice([number, templates]).format(number=number)
        return number
    
    def generate_healthcare_number():
        """건강보험번호 생성 함수"""
        #return HealthcareNumber.getContext(fake.bban())
        return fake.bban()

# PII 데이터 치환을 위한 매핑
pii_mapping = {
    'FULL_NAME': fake.name,
    'EMAIL': Email.generate_valid_email,
    'PHONE_NUM': PhoneNumber.generate_korean_phone_number,
    'PASSPORT_NUM': Passport.generate_korean_passport_number,
    'ADDRESS': Address.generate_korean_address,
    'SOCIAL_SECURITY_NUM': KoreanRRN.generate_korean_rrn,
    'HEALTHCARE_NUM': HealthcareNumber.generate_healthcare_number,  # 의료보험 번호 (임시)
    'CARD_NUM': CreditCard.generate_korean_credit_card_number,
    'SHIPMENT_TRACKING_NUM': ShipmentTracking.generate_shipment_tracking_number,
    'DRIVERS_LICENSE_NUM': DriverLicense.generate_korean_driver_license
}

def substitute_pii(text):
    """문장에서 PII 데이터를 실제 값으로 치환"""
    for label, generator in pii_mapping.items():
        text = re.sub(r'\{' + label + r'\}', generator(), text)
    return text

def generate_augmented_data(entry, n):
    """한 개의 데이터에 대해 n개의 변형 데이터를 생성"""
    augmented_entries = []

    for _ in range(n):
        used_pii_values = {}

        # 먼저 placeholder를 수집하고 대응되는 값을 생성
        placeholders = set(re.findall(r'\{(.*?)\}', entry["sentence"]))
        for label_info in entry["labels"]:
            word = label_info["word"]
            if re.match(r'\{(.*?)\}', word):
                placeholders.add(word.strip('{}'))

        for key in placeholders:
            if key in pii_mapping:
                used_pii_values[key] = pii_mapping[key]()

        # sentence 치환
        modified_text = entry["sentence"]
        for key, val in used_pii_values.items():
            modified_text = modified_text.replace(f"{{{key}}}", val)

        # labels 치환
        modified_labels = []
        for label_info in entry["labels"]:
            word = label_info["word"]
            tag = label_info["label"]

            if re.match(r'\{(.*?)\}', word):
                pii_key = word.strip('{}')
                new_word = used_pii_values.get(pii_key, word)

                if pii_key == "ADDRESS":
                    tokens = new_word.strip().split()
                    for i, token in enumerate(tokens):
                        bio_tag = f"{'B' if i == 0 else 'I'}-ADDRESS"
                        modified_labels.append({"word": token, "label": bio_tag})
                else:
                    modified_labels.append({"word": new_word, "label": tag})
            else:
                modified_labels.append({"word": word, "label": tag})


        augmented_entries.append({
            "sentence": modified_text,
            "labels": modified_labels
        })

    return augmented_entries

def process_json(input_file, output_file, n):
    """JSON 파일을 읽고 PII 데이터를 치환 후 저장"""
    #with open(input_file, 'r', encoding='utf-8') as f:
    #    data = json.load(f)
        
    #data = data.replace('"sentences":', '"sentence":')
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()  # JSON 데이터를 문자열로 읽음

    # "sentences" → "sentence" 단순 치환
    data = data.replace('"sentences":', '"sentence":')

    # 문자열을 다시 JSON 형태로 로드
    data = json.loads(data)
    
    processed_data = []

    for entry in data:
        if isinstance(entry, list):
            for item in entry:
                for variation in generate_augmented_data(item, n):
                    processed_data.append([variation])  # Each entry as a new list
        else:
            for variation in generate_augmented_data(entry, n):
                processed_data.append([variation])  # Each entry as a new list

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="input JSON file name")
parser.add_argument("--size", type=int, default=2, help="number of variations to generate per entry")
args = parser.parse_args()


# 300개 주입 후 총 문장 개수 검토 필요함             
n_variations = args.size#1#100 # 100  # 한 문장당 100개의 랜덤 변형 생성

input_json_file = f"./dataset/{args.input_file}"   
output_json_file = f"{input_json_file[:-5]}_faker.json"

#input_json_file = f"./results/zeroshot/{args.input_file}" 
#output_json_file = f"./dataset/{args.input_file[:-5]}_faker.json"

print("Faker started")
process_json(input_json_file, output_json_file, n_variations)
print("Faker finished")