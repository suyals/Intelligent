import os
import json
import joblib
import numpy as np
from typing import Dict, List, Any
from backend.preprocessor import clean_text
from backend.schemas import PredictionResult

PRIORITY_SLA = {
    "High": "Urgent Resolution within 24 Hours (Immediate Administrative Escalation)",
    "Medium": "Standard Resolution within 48-72 Hours (Departmental Queue)",
    "Low": "Informational / Routine Response within 3-5 Working Days"
}

PRIORITY_ACTION = {
    "High": "Escalate to Head of Department / Registrar's Emergency Desk immediately.",
    "Medium": "Route to designated departmental officer for review and processing.",
    "Low": "Provide self-service information or queue for standard administrative response."
}

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.vectorizer = None
        self.priority_model = None
        self.category_model = None
        self.label_metadata = None
        self.model_metrics = None
        self.loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        p_path = os.path.join(self.models_dir, "priority_model.joblib")
        c_path = os.path.join(self.models_dir, "category_model.joblib")
        meta_path = os.path.join(self.models_dir, "label_metadata.json")
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")

        if not (os.path.exists(vec_path) and os.path.exists(p_path) and os.path.exists(c_path)):
            raise FileNotFoundError(
                f"Model artifacts not found in '{self.models_dir}'. Please run 'python scripts/train.py' first."
            )

        self.vectorizer = joblib.load(vec_path)
        self.priority_model = joblib.load(p_path)
        self.category_model = joblib.load(c_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.label_metadata = json.load(f)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.model_metrics = json.load(f)

        self.loaded = True

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract top TF-IDF keywords from the input complaint."""
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_vec = self.vectorizer.transform([text]).toarray()[0]
        nonzero_indices = np.where(tfidf_vec > 0)[0]
        if len(nonzero_indices) == 0:
            return [w for w in text.split() if len(w) > 3][:top_k]
        
        # Sort by TF-IDF weight descending
        sorted_indices = nonzero_indices[np.argsort(tfidf_vec[nonzero_indices])[::-1]]
        return [feature_names[i] for i in sorted_indices[:top_k]]

    def generate_explanation(self, category: str, priority: str, keywords: List[str]) -> Dict[str, str]:
        kw_str = ", ".join([f"'{k}'" for k in keywords]) if keywords else "general query phrasing"

        # Category explanation
        cat_explanations = {
            "Academics & Examination": f"Classified under 'Academics & Examination' due to academic terminology ({kw_str}) relating to course registration, exams, grading, or schedules.",
            "Finance & Accounts": f"Classified under 'Finance & Accounts' due to financial and billing terms ({kw_str}) involving fees, payments, bank deductions, or scholarships.",
            "Registrar & Student Records": f"Classified under 'Registrar & Student Records' due to official documentation terms ({kw_str}) regarding certificates, NOCs, ID cards, or official records.",
            "IT & Technical Support": f"Classified under 'IT & Technical Support' due to technical infrastructure terms ({kw_str}) regarding passwords, portal access, or campus Wi-Fi.",
            "Hostel & Housing": f"Classified under 'Hostel & Housing' due to residential terms ({kw_str}) concerning room allotment, hostel change, or living facilities.",
            "Campus Facilities & Transport": f"Classified under 'Campus Facilities & Transport' due to campus amenity terms ({kw_str}) regarding bus schedules, cafeteria, library, or auditorium.",
            "Student Affairs & Clubs": f"Classified under 'Student Affairs & Clubs' due to campus life terms ({kw_str}) concerning student clubs, sports, workshops, or cultural events."
        }
        cat_exp = cat_explanations.get(
            category,
            f"Classified under '{category}' based on dominant semantic features ({kw_str})."
        )

        # Priority explanation
        prio_explanations = {
            "High": f"Assigned 'High' priority because key terms ({kw_str}) indicate critical deadlines, financial deductions, exam eligibility blocks, or urgent compliance risks.",
            "Medium": f"Assigned 'Medium' priority because key terms ({kw_str}) pertain to routine administrative requests, academic modifications, or certificate applications.",
            "Low": f"Assigned 'Low' priority because key terms ({kw_str}) represent standard informational queries or general campus amenity inquiries with no urgent operational impact."
        }
        prio_exp = prio_explanations.get(
            priority,
            f"Assigned '{priority}' priority based on contextual urgency and keywords ({kw_str})."
        )

        overall = f"{cat_exp} {prio_exp}"
        return {
            "overall": overall,
            "category_explanation": cat_exp,
            "priority_explanation": prio_exp
        }

    def predict(self, raw_text: str) -> PredictionResult:
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")

        cleaned = clean_text(raw_text)
        tfidf_features = self.vectorizer.transform([cleaned])

        # Priority prediction
        p_classes = list(self.priority_model.classes_)
        if hasattr(self.priority_model, "predict_proba"):
            p_probs = self.priority_model.predict_proba(tfidf_features)[0]
            p_best_idx = int(np.argmax(p_probs))
            p_pred = p_classes[p_best_idx]
            p_conf = float(p_probs[p_best_idx])
            priority_probabilities = {cls: round(float(prob), 4) for cls, prob in zip(p_classes, p_probs)}
        else:
            p_pred = self.priority_model.predict(tfidf_features)[0]
            p_conf = 1.0
            priority_probabilities = {cls: 1.0 if cls == p_pred else 0.0 for cls in p_classes}

        # Category prediction
        c_classes = list(self.category_model.classes_)
        if hasattr(self.category_model, "predict_proba"):
            c_probs = self.category_model.predict_proba(tfidf_features)[0]
            c_best_idx = int(np.argmax(c_probs))
            c_pred = c_classes[c_best_idx]
            c_conf = float(c_probs[c_best_idx])
            category_probabilities = {cls: round(float(prob), 4) for cls, prob in zip(c_classes, c_probs)}
        else:
            c_pred = self.category_model.predict(tfidf_features)[0]
            c_conf = 1.0
            category_probabilities = {cls: 1.0 if cls == c_pred else 0.0 for cls in c_classes}

        # Keywords and explanations
        keywords = self.extract_keywords(cleaned, top_k=5)
        explanations = self.generate_explanation(c_pred, p_pred, keywords)

        return PredictionResult(
            complaint=raw_text,
            cleaned_complaint=cleaned,
            predicted_category=c_pred,
            category_confidence=round(c_conf, 4) if c_conf is not None else 1.0,
            category_probabilities=category_probabilities,
            predicted_priority=p_pred,
            priority_confidence=round(p_conf, 4),
            priority_probabilities=priority_probabilities,
            contributing_keywords=keywords,
            explanation=explanations["overall"],
            category_explanation=explanations["category_explanation"],
            priority_explanation=explanations["priority_explanation"],
            suggested_action=PRIORITY_ACTION.get(p_pred, "Review complaint through standard channels."),
            resolution_sla=PRIORITY_SLA.get(p_pred, "Standard response time applies.")
        )
