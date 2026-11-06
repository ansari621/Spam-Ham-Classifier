"""Step 1: split messages.csv into 85% training_data/ and 15% test/ (stratified),
and provide cleaning + tokenisation used by every other module.

Run once:  python src/preprocessing.py
"""
import csv
import random
import re
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = PROJECT_ROOT / "messages.csv"
TRAIN_DIR = PROJECT_ROOT / "training_data"      # training_data/ham, training_data/spam
TEST_DIR = PROJECT_ROOT / "test"                # test/email.txt, email1.txt, ...
TEST_LABELS = PROJECT_ROOT / "test_labels.csv"  # true labels of the test emails
TEST_FRACTION = 0.15
SEED = 42

# words/numbers (with optional apostrophe part) plus the spam-typical symbols $ ! %
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[$!%]")
HEADER_PATTERN = re.compile(
    r"^(subject|from|to|cc|bcc|date|reply-to|return-path|message-id|received|"
    r"mime-version|content-type|content-length|content-transfer-encoding|x-[\w-]+)\s*:.*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- cleaning / tokens
def strip_headers(text: str) -> str:
    """Drop leading header lines (From:, To:, ...). Keep the subject words, not the label 'Subject:'."""
    kept, in_header = [], True
    for line in text.splitlines():
        if in_header:
            if not line.strip():
                in_header = False
                continue
            if HEADER_PATTERN.match(line.strip()):
                if line.strip().lower().startswith("subject"):
                    kept.append(line.split(":", 1)[1])
                continue
            in_header = False
        kept.append(line)
    return "\n".join(kept)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(" ".join(strip_headers(text).lower().split()))


# ---------------------------------------------------------------- loading
def load_dataset() -> tuple[list[str], list[int]]:
    """Read messages.csv (columns: subject, message, label). label 1 = spam, 0 = ham."""
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_FILE}")
    documents, labels = [], []
    with SOURCE_FILE.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("label") or "").strip()
            if label not in {"0", "1"}:
                raise ValueError(f"Unexpected label {label!r} in messages.csv")
            parts = ((row.get("subject") or "").strip(), (row.get("message") or "").strip())
            text = "\n".join(p for p in parts if p)
            if text:
                documents.append(text)
                labels.append(int(label))
    return documents, labels


def load_training_data() -> tuple[list[str], list[int]]:
    """Read the emails written to training_data/ham and training_data/spam."""
    documents, labels = [], []
    for name, label in (("ham", 0), ("spam", 1)):
        for path in sorted((TRAIN_DIR / name).glob("*.txt")):
            documents.append(path.read_text(encoding="utf-8", errors="replace"))
            labels.append(label)
    if not documents:
        raise FileNotFoundError("training_data/ is empty. Run: python src/preprocessing.py")
    return documents, labels


def load_test_data() -> tuple[list[str], list[int], list[str]]:
    """Read test/ emails and their true labels from test_labels.csv."""
    documents, labels, names = [], [], []
    with TEST_LABELS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = TEST_DIR / row["filename"]
            documents.append(path.read_text(encoding="utf-8", errors="replace"))
            labels.append(int(row["label"]))
            names.append(row["filename"])
    return documents, labels, names


# ---------------------------------------------------------------- stratified split
def stratified_split(labels: list[int], test_fraction: float, seed: int):
    """Split indices per class, so train and test keep the same spam:ham ratio."""
    rng = random.Random(seed)
    train_idx, test_idx = [], []
    for label in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == label]
        rng.shuffle(idx)
        n_test = round(len(idx) * test_fraction)
        test_idx += idx[:n_test]
        train_idx += idx[n_test:]
    rng.shuffle(test_idx)  # mix spam and ham in test file numbering
    return sorted(train_idx), test_idx


def _clear_txt(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.txt"):
        old.unlink()


def create_split() -> None:
    documents, labels = load_dataset()
    train_idx, test_idx = stratified_split(labels, TEST_FRACTION, SEED)

    for name in ("ham", "spam"):
        _clear_txt(TRAIN_DIR / name)
    _clear_txt(TEST_DIR)

    counter = {0: 0, 1: 0}
    for i in train_idx:
        counter[labels[i]] += 1
        folder = TRAIN_DIR / ("spam" if labels[i] == 1 else "ham")
        (folder / f"email_{counter[labels[i]]:05d}.txt").write_text(documents[i], encoding="utf-8")

    with TEST_LABELS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label"])
        for n, i in enumerate(test_idx):
            name = "email.txt" if n == 0 else f"email{n}.txt"   # email.txt, email1.txt, email2.txt, ...
            (TEST_DIR / name).write_text(documents[i], encoding="utf-8")
            writer.writerow([name, labels[i]])

    def show(title, idx):
        spam = sum(labels[i] for i in idx)
        print(f"{title:9s}: {len(idx):5d} emails | spam {spam:4d} ({spam / len(idx):.2%}) | ham {len(idx) - spam:4d}")

    show("Total", range(len(labels)))
    show("Train 85%", train_idx)
    show("Test 15%", test_idx)


if __name__ == "__main__":
    create_split()
