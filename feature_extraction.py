"""Step 2: turn token lists into numeric feature vectors (bag-of-words or TF-IDF)."""
import math
from collections import Counter

SparseVector = dict[int, float]


class FeatureBuilder:
    """Learns the vocabulary (and IDF) from training emails only, then converts any email to a vector."""

    def __init__(self, feature_type: str = "tfidf", min_df: int = 2):
        if feature_type not in ("bow", "tfidf"):
            raise ValueError("feature_type must be 'bow' or 'tfidf'")
        self.feature_type = feature_type
        self.min_df = min_df          # keep words that appear in at least min_df training emails
        self.words: list[str] = []
        self.index: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, tokenized_docs: list[list[str]]) -> "FeatureBuilder":
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))
        self.words = sorted(w for w, f in doc_freq.items() if f >= self.min_df)
        self.index = {w: i for i, w in enumerate(self.words)}
        n = len(tokenized_docs)
        self.idf = [math.log((1 + n) / (1 + doc_freq[w])) + 1 for w in self.words]
        return self

    def transform_one(self, tokens: list[str]) -> SparseVector:
        counts = Counter(self.index[t] for t in tokens if t in self.index)
        if self.feature_type == "bow":
            return {i: float(c) for i, c in counts.items()}
        vec = {i: (1 + math.log(c)) * self.idf[i] for i, c in counts.items()}   # TF x IDF
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {i: v / norm for i, v in vec.items()} if norm else vec

    def transform(self, tokenized_docs: list[list[str]]) -> list[SparseVector]:
        return [self.transform_one(t) for t in tokenized_docs]

    def to_dict(self) -> dict:
        return {"feature_type": self.feature_type, "min_df": self.min_df, "words": self.words, "idf": self.idf}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureBuilder":
        fb = cls(data["feature_type"], data["min_df"])
        fb.words, fb.idf = data["words"], data["idf"]
        fb.index = {w: i for i, w in enumerate(fb.words)}
        return fb
