import os
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Canonical domain category mapping for student queries
QUERY_CATEGORY_MAP = {
    # Academics & Examination
    "I am unable to access exam hall ticket.": "Academics & Examination",
    "I cannot submit my exam form and today is the last date.": "Academics & Examination",
    "I missed my exam registration due to portal error.": "Academics & Examination",
    "The assignment portal is not working before deadline.": "Academics & Examination",
    "My result shows absent but I attended exam.": "Academics & Examination",
    "How to apply for revaluation?": "Academics & Examination",
    "Can I change my project guide?": "Academics & Examination",
    "I want to change my elective subject.": "Academics & Examination",
    "When will semester timetable be released?": "Academics & Examination",
    "Holiday list for this semester?": "Academics & Examination",

    # Finance & Accounts
    "My fee payment failed but amount deducted.": "Finance & Accounts",
    "Urgent: hostel fee payment not updated.": "Finance & Accounts",
    "My scholarship has not been credited yet.": "Finance & Accounts",

    # Registrar & Student Records
    "I need migration certificate process.": "Registrar & Student Records",
    "Request for bonafide certificate.": "Registrar & Student Records",
    "Degree certificate required urgently for job.": "Registrar & Student Records",
    "Internship deadline today, need NOC approval.": "Registrar & Student Records",
    "Please update my contact number.": "Registrar & Student Records",
    "Where to collect ID card?": "Registrar & Student Records",

    # IT & Technical Support
    "WiFi password for campus?": "IT & Technical Support",
    "How to reset student portal password?": "IT & Technical Support",

    # Hostel & Housing
    "How can I apply for hostel change?": "Hostel & Housing",

    # Campus Facilities & Transport
    "Where is the auditorium located?": "Campus Facilities & Transport",
    "When does cafeteria open?": "Campus Facilities & Transport",
    "Bus schedule for weekend?": "Campus Facilities & Transport",
    "What are library timings?": "Campus Facilities & Transport",

    # Student Affairs & Clubs
    "Any cultural events this week?": "Student Affairs & Clubs",
    "How to join sports club?": "Student Affairs & Clubs",
    "I want to join a club in university.": "Student Affairs & Clubs",
    "Is there any AI workshop this month?": "Student Affairs & Clubs",
}

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
    """Preprocess and clean student complaint text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
    # Remove special punctuation but preserve word characters and whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_evaluate():
    print("=" * 70)
    print("INTELLIGENT STUDENT COMPLAINT CLASSIFIER & PRIORITY PREDICTOR")
    print("End-to-End Training & Evaluation Pipeline")
    print("=" * 70)

    train_path = os.path.join("data", "student_query_train_5000.csv")
    test_path = os.path.join("data", "student_query_test_1000.csv")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n1. Loading datasets from '{train_path}' and '{test_path}'...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"   Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    # Add canonical Category column
    train_df["Category"] = train_df["Student_Query"].map(QUERY_CATEGORY_MAP)
    test_df["Category"] = test_df["Student_Query"].map(QUERY_CATEGORY_MAP)

    # Clean text
    print("\n2. Performing NLP Text Preprocessing (cleaning, token normalization)...")
    train_df["cleaned_text"] = train_df["Student_Query"].apply(clean_text)
    test_df["cleaned_text"] = test_df["Student_Query"].apply(clean_text)

    # Feature extraction with TF-IDF
    print("\n3. Vectorizing complaints using TF-IDF (unigrams + bigrams, sublinear TF)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_test = vectorizer.transform(test_df["cleaned_text"])
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)} features")

    # Define candidate classifiers
    candidate_models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Calibrated Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    }

    targets = {
        "Priority": {
            "y_train": train_df["Priority"].values,
            "y_test": test_df["Priority"].values,
            "classes": sorted(list(train_df["Priority"].unique()))
        },
        "Category": {
            "y_train": train_df["Category"].values,
            "y_test": test_df["Category"].values,
            "classes": sorted(list(train_df["Category"].unique()))
        }
    }

    model_comparison_results = {}
    best_models = {}
    final_evaluations = {}
    confusion_matrices = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for target_name, target_info in targets.items():
        print(f"\n" + "-" * 60, flush=True)
        print(f"EVALUATING MODELS FOR TARGET: [{target_name.upper()}]", flush=True)
        print(f"Classes ({len(target_info['classes'])}): {target_info['classes']}", flush=True)
        print("-" * 60, flush=True)

        y_train = target_info["y_train"]
        y_test = target_info["y_test"]
        target_comparison = {}
        best_score = -1.0
        best_name = None

        for name, model in candidate_models.items():
            print(f"\n   Running 5-Fold Cross-Validation for '{name}'...", flush=True)
            scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
            cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

            mean_acc = float(np.mean(cv_results["test_accuracy"]))
            mean_prec = float(np.mean(cv_results["test_precision_weighted"]))
            mean_rec = float(np.mean(cv_results["test_recall_weighted"]))
            mean_f1 = float(np.mean(cv_results["test_f1_weighted"]))

            print(f"     -> CV Accuracy: {mean_acc * 100:.2f}% | F1-Score: {mean_f1 * 100:.2f}%")

            target_comparison[name] = {
                "cv_accuracy": round(mean_acc, 4),
                "cv_precision": round(mean_prec, 4),
                "cv_recall": round(mean_rec, 4),
                "cv_f1": round(mean_f1, 4)
            }

            if mean_f1 > best_score:
                best_score = mean_f1
                best_name = name

        print(f"\n   [WINNER for {target_name}]: '{best_name}' (CV F1: {best_score * 100:.2f}%)")
        model_comparison_results[target_name] = {
            "candidates": target_comparison,
            "selected_model": best_name
        }

        # Train best model on full train set and evaluate on test set
        best_clf = clone(candidate_models[best_name])
        best_clf.fit(X_train, y_train)
        best_models[target_name] = best_clf

        y_pred = best_clf.predict(X_test)

        test_acc = float(accuracy_score(y_test, y_pred))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

        clf_rep = classification_report(y_test, y_pred, target_names=target_info["classes"], output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=target_info["classes"])

        print(f"\n   Final Test Set Performance for {target_name} ({best_name}):")
        print(f"     Test Accuracy:        {test_acc * 100:.2f}%")
        print(f"     Test Precision (wt):  {p_weighted * 100:.2f}%")
        print(f"     Test Recall (wt):     {r_weighted * 100:.2f}%")
        print(f"     Test F1-Score (wt):   {f1_weighted * 100:.2f}%")
        print(f"     Test F1-Score (macro):{f1_macro * 100:.2f}%")

        final_evaluations[target_name] = {
            "model_name": best_name,
            "test_accuracy": round(test_acc, 4),
            "test_precision_weighted": round(float(p_weighted), 4),
            "test_recall_weighted": round(float(r_weighted), 4),
            "test_f1_weighted": round(float(f1_weighted), 4),
            "test_f1_macro": round(float(f1_macro), 4),
            "class_metrics": {k: v for k, v in clf_rep.items() if k in target_info["classes"]},
            "classes": target_info["classes"]
        }

        confusion_matrices[target_name] = {
            "labels": target_info["classes"],
            "matrix": cm.tolist()
        }

    # Save artifacts
    print("\n" + "=" * 70)
    print("4. Saving Model Artifacts and Preprocessing Pipeline...")
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(best_models["Priority"], os.path.join(models_dir, "priority_model.joblib"))
    joblib.dump(best_models["Category"], os.path.join(models_dir, "category_model.joblib"))

    label_metadata = {
        "priority_classes": targets["Priority"]["classes"],
        "category_classes": targets["Category"]["classes"],
        "query_category_map": QUERY_CATEGORY_MAP
    }
    with open(os.path.join(models_dir, "label_metadata.json"), "w") as f:
        json.dump(label_metadata, f, indent=2)

    # Save metrics and confusion matrix
    analytics_payload = {
        "dataset_summary": {
            "train_total": len(train_df),
            "test_total": len(test_df),
            "priority_distribution_train": train_df["Priority"].value_counts().to_dict(),
            "priority_distribution_test": test_df["Priority"].value_counts().to_dict(),
            "category_distribution_train": train_df["Category"].value_counts().to_dict(),
            "category_distribution_test": test_df["Category"].value_counts().to_dict(),
            "vocabulary_features": len(vectorizer.vocabulary_)
        },
        "model_comparison": model_comparison_results,
        "final_evaluations": final_evaluations,
        "confusion_matrices": confusion_matrices
    }

    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(analytics_payload, f, indent=2)

    with open(os.path.join(models_dir, "confusion_matrices.json"), "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    print("   Artifacts successfully written to:")
    print("     - models/tfidf_vectorizer.joblib")
    print("     - models/priority_model.joblib")
    print("     - models/category_model.joblib")
    print("     - models/label_metadata.json")
    print("     - models/model_metrics.json")
    print("     - models/confusion_matrices.json")
    print("=" * 70)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate()
