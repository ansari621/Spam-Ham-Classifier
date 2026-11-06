"""Step 5: classify every email in the ./test folder -> 1 (spam) or 0 (not spam).

Run:  python src/predict.py
Or:   from predict import predict_emails ; predict_emails()
"""
import json
import re
import sys
from pathlib import Path

from feature_extraction import FeatureBuilder
from naive_bayes import MultinomialNaiveBayes
from preprocessing import PROJECT_ROOT, tokenize

MODEL_FILE = PROJECT_ROOT / "model" / "model.json"


def load_model():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"{MODEL_FILE} not found. Run: python src/train.py")
    data = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    return FeatureBuilder.from_dict(data["features"]), MultinomialNaiveBayes.from_dict(data["classifier"])


def _natural_order(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def predict_emails(folder=None) -> list[tuple[str, int]]:
    """Read every email*.txt in the test folder (current directory by default) -> [(name, 1 or 0)]."""
    if folder is None:
        folder = Path.cwd() / "test"
        if not folder.is_dir():
            folder = PROJECT_ROOT / "test"
    files = sorted(Path(folder).glob("email*.txt"), key=_natural_order)
    builder, model = load_model()
    return [(f.name, model.predict_one(builder.transform_one(tokenize(f.read_text(encoding="utf-8", errors="replace")))))
            for f in files]


def main() -> None:
    results = predict_emails(sys.argv[1] if len(sys.argv) > 1 else None)
    if not results:
        print("No email*.txt files found in the test folder.")
    for name, label in results:
        print(f"{name}: {label}")


if __name__ == "__main__":
    main()
