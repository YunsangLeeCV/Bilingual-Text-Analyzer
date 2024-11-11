import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from konlpy.tag import Okt
from collections import Counter

# WordNet 품사 태그와 WordNetLemmatizer 품사 태그 간 매핑
TAG_MAP = {
    'N': wordnet.NOUN,
    'V': wordnet.VERB,
    'R': wordnet.ADV,
    'J': wordnet.ADJ
}

def preprocess_text_en(file_path, pattern, stop_words):
    """
    영어 텍스트 파일을 정리하고 불용어 제거 및 표제어 추출을 수행합니다.
    """
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    all_tokens = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = re.sub(pattern, "", line).strip().lower()
            tokens = tokenizer.tokenize(cleaned_line)
            for token, tag in nltk.pos_tag(tokens):
                if token not in stop_words:
                    wn_tag = TAG_MAP.get(tag[0].upper(), wordnet.NOUN)
                    lemmatized_token = lemmatizer.lemmatize(token, wn_tag)
                    all_tokens.append(lemmatized_token)
    
    return all_tokens

def preprocess_text_ko(file_path, pattern, stop_words):
    """
    한국어 텍스트 파일을 정리하고 불용어 제거 및 품사 태깅을 수행합니다.
    """
    okt = Okt()
    tagged_tokens = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = re.sub(pattern, "", line).strip().lower()
            tokens = okt.pos(cleaned_line, norm=True, stem=True)
            filtered_tokens = [(word, tag) for word, tag in tokens if word not in stop_words]
            tagged_tokens.extend(filtered_tokens)
  
    return tagged_tokens

def get_top_n_words_with_tags(tokens, n=10):
    """
    품사 태그가 포함된 토큰 리스트에서 상위 N개의 빈도 높은 단어와 품사 반환.
    """
    counter = Counter(tokens)
    top_n = counter.most_common(n)
    return [(word, tag, freq) for ((word, tag), freq) in top_n]

def get_top_n_words_excluding_josa(tokens, n=10):
    """
    조사(Josa)를 제외한 상위 N개의 빈도 높은 단어와 품사 반환.
    """
    filtered_tokens = [token for token in tokens if token[1] != 'Josa']
    counter = Counter(filtered_tokens)
    top_n = counter.most_common(n)
    return [(word, tag, freq) for ((word, tag), freq) in top_n]

def get_top_n_words(tokens, n=10):
    """
    토큰 리스트에서 상위 N개의 빈도 높은 단어 반환.
    """
    return pd.Series(tokens).value_counts().head(n)

# 정규 표현식 패턴 정의
pattern = r"[^ ㄱ-ㅣ가-힣a-zA-Z]+"  # 한글 및 알파벳 이외의 문자 제거

# 파일 경로 정의
EN_path = r"C:\Users\nana\Documents\Python\Steve jobs 2005 Commencement Address.txt"
KO_path = r"C:\Users\nana\Documents\Python\Steve jobs 2005 Commencement Address_Korean.txt"
KO_Stop_Words_Path = r"C:\Users\nana\Documents\Python\stopwords_korean.txt"

# 불용어 로드
stop_words_eng = set(stopwords.words('english'))
with open(KO_Stop_Words_Path, "r", encoding="utf-8") as f:
    stop_words_ko = set(f.read().split())

# 영어 및 한국어 텍스트 전처리 및 빈도 계산
english_tokens = preprocess_text_en(EN_path, pattern, stop_words_eng)
korean_tagged_tokens = preprocess_text_ko(KO_path, pattern, stop_words_ko)

# 상위 10개 단어 (영어)
top_english_words = get_top_n_words(english_tokens)

# 상위 10개 단어 (조사 포함, 한국어)
top_korean_words_with_tags = get_top_n_words_with_tags(korean_tagged_tokens)

# 조사 제외한 상위 10개 단어 (한국어)
top_korean_words_excluding_josa = get_top_n_words_excluding_josa(korean_tagged_tokens)

# 결과 출력
print("Top 10 English words")
print(top_english_words)

print("\nTop 10 Korean words with POS tags (including Josa)")
for word, tag, freq in top_korean_words_with_tags:
    print(f"{word} ({tag}): {freq} Times")

print("\nTop 10 Korean words with POS tags (excluding Josa)")
for word, tag, freq in top_korean_words_excluding_josa:
    print(f"{word} ({tag}): {freq} Times")