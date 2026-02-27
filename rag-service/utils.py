# utils.py
import re

RELEVANCE_THRESHOLD = 0.25

def faiss_score_to_cosine_sim(score: float) -> float:
    return max(0.0, 1.0 - score / 2.0)

def compute_confidence(faiss_scores: list[float]) -> float:
    if not faiss_scores:
        return 0.0
    top_scores = sorted(faiss_scores)[:3]
    similarities = [faiss_score_to_cosine_sim(s) for s in top_scores]
    avg_sim = sum(similarities) / len(similarities)
    return round(float(avg_sim * 100), 1)

def normalize_spaced_text(text: str) -> str:
    def fix_spaced_word(match):
        return match.group(0).replace(" ", "")
    pattern = r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b"
    return re.sub(pattern, fix_spaced_word, text)

def normalize_answer(text: str) -> str:
    text = normalize_spaced_text(text)
    text = re.sub(r'^(Final Answer:|Context:|Question:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()