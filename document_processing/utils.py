import re


def preprocess_text(text: str) -> str:
    """Normalize units and remove soft hyphen breaks."""
    if not text:
        return text
    # Normalize mg/l and N/mm² with optional spaces or narrow spaces
    text = re.sub(r"mg\s*/\s*l", "mg/l", text, flags=re.IGNORECASE)
    text = re.sub(r"N\s*/\s*mm²", "N/mm²", text)

    # Remove soft hyphen or hyphen at line breaks
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n\s*", "", text)

    return text
