"""

"""

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from unidecode import unidecode
import re
from functools import lru_cache
from icecream import ic


lemmatizer = WordNetLemmatizer(); lemmatizer.lemmatize('')  # loads lemmatizer
punc_ptn = re.compile(r'\W', re.ASCII)
stops = set(stopwords.words('english'))


@lru_cache(maxsize=2**16)
def lemmatize(token: str) -> str:
    return lemmatizer.lemmatize(token)


def valid_token(token: str) -> bool:
    return token and token not in stops


def normalize(token: str) -> str:
    return lemmatize(token.strip().lower())


def resolve_token(token: str) -> str:
    return normal if valid_token(normal := normalize(token)) else None


def preprocess(text: str) -> list[str]:
    return [resolve_token(token) for token in re.sub(punc_ptn, ' ', unidecode(text)).split()]