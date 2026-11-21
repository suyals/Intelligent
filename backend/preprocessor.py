import re

CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bi'm\b": "i am",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bit's\b": "it is",
    r"\bwhat's\b": "what is",
    r"\bthere's\b": "there is",
}

def clean_text(text: str) -> str:
    """Clean and normalize student complaint text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
    # Replace non-alphanumeric characters with spaces (preserving letters and digits)
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text).strip()
    return text

