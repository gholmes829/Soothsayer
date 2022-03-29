"""

"""

from nltk.stem import WordNetLemmatizer
from functools import lru_cache
from nltk.corpus import stopwords
import unidecode
import re
from icecream import ic


lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))
punc_ptn = re.compile(r'\W', re.ASCII)

@lru_cache(maxsize=50000)
def lemmatize(token: str) -> str:
    return lemmatizer.lemmatize(token)
lemmatize('')


def valid_token(token: str) -> bool:
    return token and token not in stops


def normalize(token: str) -> str:
    return lemmatize(token.strip().lower())


def preprocess(text: str) -> list[str]:
    return [normal for token in re.sub(punc_ptn, ' ', unidecode.unidecode(text)).split() if valid_token(normal := normalize(token))]
