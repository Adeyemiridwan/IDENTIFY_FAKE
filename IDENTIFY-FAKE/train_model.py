import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# =========================
# LOAD DATA
# =========================
def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    return data


def prepare_data(data):
    X = data["text"]
    y = data["label"]
    return X, y


# =========================
# SPLIT DATA
# =========================
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


# =========================
# VECTORIZE
# =========================
def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        stop_words="english",
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return vectorizer, X_train_vec, X_test_vec


# =========================
# TRAIN MODELS
# =========================
def train_models(X_train_vec, y_train):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM": LinearSVC(class_weight="balanced")
    }

    for name, model in models.items():
        model.fit(X_train_vec, y_train)

    return models


# =========================
# EVALUATE MODELS
# =========================
def evaluate_models(models, X_test_vec, y_test):

    results = []

    print("\n--- MODEL COMPARISON ---")

    for name, model in models.items():

        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label="fake")
        recall = recall_score(y_test, preds, pos_label="fake")
        f1 = f1_score(y_test, preds, pos_label="fake")

        # AUC only works if probability exists
        try:
            probs = model.predict_proba(X_test_vec)[:, 1]
            auc = roc_auc_score((y_test == "fake").astype(int), probs)
        except:
            auc = 0  # SVM fallback

        print(f"\n{name}")
        print("Accuracy:", round(acc * 100, 2), "%")
        print("Precision:", round(precision * 100, 2), "%")
        print("Recall:", round(recall * 100, 2), "%")
        print("F1 Score:", round(f1 * 100, 2), "%")
        print("AUC-ROC:", round(auc * 100, 2), "%")

        results.append((name, model, f1))

    # Select best based on F1 Score
    best_model = sorted(results, key=lambda x: x[2], reverse=True)[0]

    print("\n🏆 BEST MODEL:", best_model[0])

    return best_model[1]


# =========================
# SAVE MODEL
# =========================
def save_model(model, vectorizer):
    pickle.dump(model, open("app/model/model.pkl", "wb"))
    pickle.dump(vectorizer, open("app/model/vectorizer.pkl", "wb"))
    print("\nModel saved successfully!")


# =========================
# MAIN
# =========================
def main():
    data = load_data("app/data/dataset.csv")
    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    vectorizer, X_train_vec, X_test_vec = vectorize(X_train, X_test)

    models = train_models(X_train_vec, y_train)

    best_model = evaluate_models(models, X_test_vec, y_test)

    save_model(best_model, vectorizer)


if __name__ == "__main__":
    main()