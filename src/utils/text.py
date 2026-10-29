import re

_WORD_RE = re.compile(r"\b\w+\b")
_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize text the same way preprocessor.py does before training."""
    text = str(text).lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Same tokenizer used by train_word2vec.py / train_fasttext.py."""
    return _WORD_RE.findall(str(text).lower())


def preprocess_query(query: str) -> list[str]:
    return tokenize(clean_text(query))
