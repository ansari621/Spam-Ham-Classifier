"""Step 4: tune hyper-parameters with 5-fold CV on training_data/, train the final model, test on test/.

Run:  python src/train.py
"""
import json
import random
import statistics

from evaluation import classification_metrics
from feature_extraction import FeatureBuilder
from naive_bayes import MultinomialNaiveBayes
from preprocessing import (PROJECT_ROOT, SEED, TRAIN_DIR, create_split, load_test_data,
                           load_training_data, tokenize)

MODEL_FILE = PROJECT_ROOT / "model" / "model.json"
FEATURE_TYPES = ["bow", "tfidf"]
ALPHAS = [0.01, 0.1, 0.5, 1.0]
MIN_DF = 2
K = 5


def make_folds(labels, k, seed):
    """Stratified k folds: each fold keeps the spam:ham ratio."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for label in (0, 1):
        idx = [i for i, y in enumerate(labels) if y == label]
        rng.shuffle(idx)
        for pos, i in enumerate(idx):
            folds[pos % k].append(i)
    return folds


def fit_model(feature_type, alpha, tokens, labels):
    builder = FeatureBuilder(feature_type, MIN_DF).fit(tokens)
    model = MultinomialNaiveBayes(alpha).fit(builder.transform(tokens), labels, len(builder.words))
    return builder, model


def cross_validate(feature_type, alpha, tokens, labels, folds):
    accuracies = []
    for held in folds:
        held_set = set(held)
        train = [i for i in range(len(tokens)) if i not in held_set]
        builder, model = fit_model(feature_type, alpha, [tokens[i] for i in train], [labels[i] for i in train])
        preds = model.predict(builder.transform([tokens[i] for i in held]))
        accuracies.append(classification_metrics([labels[i] for i in held], preds)["accuracy"])
    return statistics.mean(accuracies), statistics.pstdev(accuracies)


def main() -> None:
    if not TRAIN_DIR.is_dir():
        create_split()
    docs, labels = load_training_data()
    tokens = [tokenize(d) for d in docs]
    print(f"Training emails: {len(docs)} (spam {sum(labels)}, ham {len(labels) - sum(labels)})")

    folds = make_folds(labels, K, SEED)
    best = None
    print(f"\n{K}-fold cross-validation")
    for feature_type in FEATURE_TYPES:
        for alpha in ALPHAS:
            mean_acc, std = cross_validate(feature_type, alpha, tokens, labels, folds)
            print(f"  features={feature_type:5s} alpha={alpha:<5} accuracy={mean_acc:.4f} (+-{std:.4f})")
            if best is None or mean_acc > best[0]:
                best = (mean_acc, feature_type, alpha)
    _, feature_type, alpha = best
    print(f"\nBest: features={feature_type}, alpha={alpha}, CV accuracy={best[0]:.4f}")

    builder, model = fit_model(feature_type, alpha, tokens, labels)   # final fit on all training data
    MODEL_FILE.parent.mkdir(exist_ok=True)
    MODEL_FILE.write_text(json.dumps({"features": builder.to_dict(), "classifier": model.to_dict()},
                                     separators=(",", ":")), encoding="utf-8")
    print(f"Model saved to {MODEL_FILE}")

    test_docs, test_labels, _ = load_test_data()
    preds = model.predict(builder.transform([tokenize(d) for d in test_docs]))
    m = classification_metrics(test_labels, preds)
    print(f"\nTest set ({len(test_docs)} emails, never used for training/tuning)")
    print(f"  accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")
    print(f"  confusion: TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}")


if __name__ == "__main__":
    main()
