"""Step 3: Multinomial Naive Bayes with Laplace smoothing (from scratch)."""
import math

from feature_extraction import SparseVector


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.log_prior: dict[int, float] = {}
        self.log_likelihood: dict[int, list[float]] = {}

    def fit(self, vectors: list[SparseVector], labels: list[int], vocab_size: int):
        n_docs = {0: 0, 1: 0}
        counts = {0: [0.0] * vocab_size, 1: [0.0] * vocab_size}
        totals = {0: 0.0, 1: 0.0}
        for vec, y in zip(vectors, labels):
            n_docs[y] += 1
            for j, value in vec.items():
                counts[y][j] += value
                totals[y] += value
        for y in (0, 1):
            self.log_prior[y] = math.log(n_docs[y] / len(labels))
            denom = totals[y] + self.alpha * vocab_size
            self.log_likelihood[y] = [math.log((c + self.alpha) / denom) for c in counts[y]]
        return self

    def predict_one(self, vec: SparseVector) -> int:
        scores = {y: self.log_prior[y] + sum(v * self.log_likelihood[y][j] for j, v in vec.items())
                  for y in (0, 1)}
        return 1 if scores[1] > scores[0] else 0     # 1 = spam, 0 = ham

    def predict(self, vectors: list[SparseVector]) -> list[int]:
        return [self.predict_one(v) for v in vectors]

    def to_dict(self) -> dict:
        return {"alpha": self.alpha,
                "log_prior": {str(k): v for k, v in self.log_prior.items()},
                "log_likelihood": {str(k): v for k, v in self.log_likelihood.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "MultinomialNaiveBayes":
        model = cls(data["alpha"])
        model.log_prior = {int(k): v for k, v in data["log_prior"].items()}
        model.log_likelihood = {int(k): v for k, v in data["log_likelihood"].items()}
        return model
