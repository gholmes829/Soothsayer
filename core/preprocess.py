"""

"""

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from unidecode import unidecode
import re
from functools import lru_cache
from icecream import ic


stemmer = PorterStemmer(); stemmer.stem('')  # loads stem
punc_ptn = re.compile(r'\W', re.ASCII)
stops = set(stopwords.words('english'))


@lru_cache(maxsize=2**16)
def stem(token: str) -> str:
    return stemmer.stem(token)


def valid_token(token: str) -> bool:
    return token and token not in stops


def normalize(token: str) -> str:
    return stem(token.strip().lower())


def resolve_token(token: str) -> str:
    return normal if valid_token(normal := normalize(token)) else None


def preprocess(text: str) -> list[str]:
    return [resolve_token(token) for token in re.sub(punc_ptn, ' ', unidecode(text)).split()]