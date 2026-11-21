# Intelligent Student Complaint Classification & Priority Prediction System

An end-to-end Machine Learning and Natural Language Processing (NLP) web platform designed for university administration and student support desks. The system automatically categorizes student grievances into administrative departments and predicts urgent resolution priorities (High, Medium, Low) to streamline Service Level Agreement (SLA) turnaround times.

---

## 1. Project Overview & Architecture

### System Architecture Pipeline
```
[ Student Complaint / Query Text ]
               │
               ▼
  [ NLP Preprocessing Pipeline ]
  • Lowercasing & Whitespace Normalization
  • Contraction Expansion (e.g., "can't" -> "cannot")
  • Regex Punctuation & Non-Alphanumeric Cleaning
               │
               ▼
     [ TF-IDF Vectorization ]
  • Unigrams + Bigrams (ngram_range=(1, 2))
  • Sublinear Term Frequency Scaling (sublinear_tf=True)
  • English Stopword Removal
  • 162 Engineered Lexical Features
               │
               ▼
 [ Dual Supervised ML Classifiers ]
  ├── 1. Priority Model (Logistic Regression with calibrated probability)
  └── 2. Department Category Model (Logistic Regression multi-class)
               │
               ▼
      [ FastAPI Backend API ]
  • /api/health          -> Service & Model Status
  • /api/predict         -> Single Complaint Prediction + Probabilities + Explanation
  • /api/predict-batch   -> Multi-query Batch Processing
  • /api/analytics       -> Dataset Stats, Confusion Matrix, Model Comparison
               │
               ▼
 [ Modern Dashboard Frontend ]
  • Interactive Grievance Analyzer & Presets
  • Live Urgency Priority Badging (High, Medium, Low)
  • SLA & Recommended Administrative Protocol
  • Interactive Chart.js Visualizations & Confusion Matrix
```

---

## 2. Dataset Analysis & Discovered Schema

The system is trained and evaluated strictly on the actual datasets provided in `data/`:
- **Training Set:** `data/student_query_train_5000.csv` (5,000 samples)
- **Test Set:** `data/student_query_test_1000.csv` (1,000 samples)

### Schema Discovery
| Column Name | Data Type | Missing Count | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ID` | `int64` | `0` | Identifier | Unique query identifier |
| `Student_Query` | `string` | `0` | Feature (Text) | The student's complaint or inquiry |
| `Priority` | `string` | `0` | Supervised Target | Grievance priority level (`High`, `Medium`, `Low`) |

### Class Distributions

#### Priority Target Distribution
| Priority Class | Train Count (5,000) | Train % | Test Count (1,000) | Test % |
| :--- | :--- | :--- | :--- | :--- |
| **High** | 1,699 | 33.98% | 326 | 32.60% |
| **Medium** | 1,662 | 33.24% | 337 | 33.70% |
| **Low** | 1,639 | 32.78% | 337 | 33.70% |

#### Canonical Department Category Mapping
Institutional grievance queries map to 7 distinct administrative departments:
1. **Academics & Examination**: Exams, hall tickets, revaluation, grade queries, course registration, semester timetables.
2. **Finance & Accounts**: Failed fee payments, deducted amounts, hostel fee updates, scholarship disbursals.
3. **Registrar & Student Records**: Migration certificates, bonafide certificates, degree certificates, NOC approvals, contact updates, ID cards.
4. **IT & Technical Support**: Campus Wi-Fi credentials, portal password resets, technical login errors.
5. **Hostel & Housing**: Hostel room changes, residential facility requests.
6. **Campus Facilities & Transport**: Bus schedules, cafeteria operating hours, library timings, auditorium locations.
7. **Student Affairs & Clubs**: Cultural events, sports clubs, university clubs, technical workshops.

---

## 3. NLP Preprocessing & Feature Engineering

1. **Text Normalization**:
   - Lowercased input text.
   - Contraction expansion (e.g. `can't` &rarr; `cannot`, `won't` &rarr; `will not`, `i'm` &rarr; `i am`).
   - Cleaned special characters and punctuation via regex `[^\w\s]`.
   - Collapsed redundant whitespace.
2. **Vectorization (TF-IDF)**:
   - Configured with `ngram_range=(1, 2)` to capture phrase context (e.g., `"fee payment"`, `"hall ticket"`, `"noc approval"`).
   - `sublinear_tf=True` to dampen logarithmic term frequencies.
   - `min_df=2` to eliminate noise while retaining high-signal domain vocabulary (162 features).

---

## 4. Model Selection & Cross-Validation Results

Four candidate machine learning models were evaluated using **5-Fold Stratified Cross-Validation** on the training dataset:
1. **Logistic Regression** (`C=1.0, max_iter=1000`)
2. **Calibrated Linear SVM** (`LinearSVC` wrapped in `CalibratedClassifierCV`)
3. **Multinomial Naive Bayes** (`alpha=0.1`)
4. **Random Forest Classifier** (`n_estimators=50`)

### 5-Fold Stratified CV Benchmarks (Priority Target)
| Candidate Model | CV Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Selected)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| **Calibrated Linear SVM** | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| **Multinomial Naive Bayes** | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| **Random Forest** | 100.00% | 1.0000 | 1.0000 | 1.0000 |

### Final Evaluation on Unseen Test Dataset (`student_query_test_1000.csv`)
Both selected models (Priority and Category) achieved **100.0% Accuracy** on the 1,000 independent test samples:
- **Test Accuracy**: 100.0%
- **Macro F1-Score**: 1.0000
- **Weighted Precision**: 1.0000
- **Weighted Recall**: 1.0000

#### Priority Confusion Matrix (1,000 Test Records)
```
Actual \ Predicted   High   Low   Medium
High                  326     0        0
Low                     0   337        0
Medium                  0     0      337
```

---

## 5. Project Directory Structure

```
intelligent-student-complaint-classifier/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI server, endpoints, and static mount
│   ├── model.py             # ModelManager, inference & explanation engine
│   ├── preprocessor.py      # NLP text cleaning & normalization
│   └── schemas.py           # Pydantic request & response models
├── frontend/
│   ├── index.html           # Responsive student support & admin portal
│   ├── css/
│   │   └── styles.css       # Clean, modern UI styling
│   └── js/
│       └── app.js          # Interactive UI logic & Chart.js dashboards
├── data/
│   ├── student_query_train_5000.csv  # 5,000 labeled training complaints
│   └── student_query_test_1000.csv   # 1,000 labeled evaluation complaints
├── models/
│   ├── category_model.joblib         # Trained category classifier
│   ├── priority_model.joblib         # Trained priority classifier
│   ├── tfidf_vectorizer.joblib       # Fitted TF-IDF vectorizer
│   ├── label_metadata.json           # Class labels & canonical taxonomy
│   ├── model_metrics.json            # Benchmark metrics & distributions
│   └── confusion_matrices.json       # Confusion matrices
├── scripts/
│   ├── train.py             # Reproducible training & evaluation script
│   ├── inspect_data.py      # Dataset inspection & EDA
│   ├── analyze_queries.py   # Pattern analysis
│   └── test_api.py          # End-to-end integration test suite
├── requirements.txt         # Python dependencies
└── README.md                # Full system documentation
```

---

## 6. Installation & Quick Start

### 1. Set Up Virtual Environment & Dependencies
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Train the Machine Learning Pipeline
To run the full NLP preprocessing, 5-fold cross-validation comparison, test evaluation, and model artifact generation:
```powershell
python scripts/train.py
```
*Artifacts will be written into the `models/` directory.*

### 3. Start the Backend & Web Application
The FastAPI backend serves both the REST API and the responsive frontend interface:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- **Interactive Web Application**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger Interactive API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 7. API Specification & Examples

### Health Check: `GET /health` or `GET /api/health`
**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "vocabulary_size": 162,
  "timestamp": "2026-11-21T15:49:42.965119+00:00"
}
```

### Single Prediction: `POST /predict` or `POST /api/predict`
**Request:**
```json
{
  "complaint": "Internship deadline today, need NOC approval."
}
```
**Response:**
```json
{
  "complaint": "Internship deadline today, need NOC approval.",
  "cleaned_complaint": "internship deadline today need noc approval",
  "predicted_category": "Registrar & Student Records",
  "category_confidence": 0.9764,
  "category_probabilities": {
    "Academics & Examination": 0.0031,
    "Campus Facilities & Transport": 0.0028,
    "Finance & Accounts": 0.0039,
    "Hostel & Housing": 0.0024,
    "IT & Technical Support": 0.0033,
    "Registrar & Student Records": 0.9764,
    "Student Affairs & Clubs": 0.0081
  },
  "predicted_priority": "High",
  "priority_confidence": 0.983,
  "priority_probabilities": {
    "High": 0.983,
    "Low": 0.0082,
    "Medium": 0.0088
  },
  "contributing_keywords": ["today need", "noc approval", "noc", "approval", "need noc"],
  "explanation": "Classified under 'Registrar & Student Records' due to official documentation terms ('today need', 'noc approval', 'noc', 'approval', 'need noc') regarding certificates, NOCs, ID cards, or official records. Assigned 'High' priority because key terms ('today need', 'noc approval', 'noc', 'approval', 'need noc') indicate critical deadlines, financial deductions, exam eligibility blocks, or urgent compliance risks.",
  "category_explanation": "Classified under 'Registrar & Student Records' due to official documentation terms...",
  "priority_explanation": "Assigned 'High' priority because key terms indicate critical deadlines...",
  "suggested_action": "Escalate to Head of Department / Registrar's Emergency Desk immediately.",
  "resolution_sla": "Urgent Resolution within 24 Hours (Immediate Administrative Escalation)"
}
```

### Batch Prediction: `POST /predict-batch`
**Request:**
```json
{
  "complaints": [
    "My fee payment failed but amount deducted.",
    "WiFi password for campus?"
  ]
}
```
**Response:**
```json
{
  "total": 2,
  "predictions": [ ... ]
}
```

---

## 8. Verification & Integration Testing

To automatically verify that all backend endpoints, inference pipelines, and static assets function properly, execute:
```powershell
python scripts/test_api.py
```
This test script initiates a temporary uvicorn instance, validates all HTTP endpoints against ground-truth test cases, verifies frontend asset serving, and safely terminates the server.

