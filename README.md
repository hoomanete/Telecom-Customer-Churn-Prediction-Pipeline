# Telecom-Customer-Churn-Prediction-Pipeline

# End-to-End Telecom Customer Churn Pipeline (GitHub Project)

Build a **Telecom Customer Churn Prediction Pipeline** using the IBM “Telco Customer Churn” dataset on Kaggle and implement all stages from raw data to deployment and monitoring.[web:44][web:56]

---

## 1. Project overview

The goal is to design a production‑style, end‑to‑end machine learning system:

- Start from raw CSV churn data (customer demographics, contracts, services, billing, and `Churn` label).[web:44][web:56]  
- Build an automated pipeline: ingestion → validation → feature engineering → training → evaluation → explanation → serving → monitoring.  
- Package everything with a clean repo structure, Docker, and CI so it is easy to run from GitHub.

---

## 2. Data source

Use the **IBM Telco Customer Churn dataset (Kaggle)**.

- The dataset, originally supplied by IBM, has ~7,043 customers and 20+ attributes, including tenure, contract type, monthly charges, total charges, and a binary churn flag.[web:44][web:54][web:58]  
- Many studies use this dataset for churn prediction with models such as logistic regression, Random Forest, XGBoost, and deep learning methods (FT‑Transformer, TabNet, hybrid CNN/BiLSTM, etc.).[web:44][web:54][web:57][web:61]  
- Example academic references using this dataset:  
  - Telco churn prediction with various ML algorithms (Random Forest, XGBoost, etc.).[web:48][web:58][web:63]  
  - Advanced stacked/ensemble models with SHAP explanations on IBM Telco.[web:55][web:63]  

In your README, add a **Dataset** section:

> This project uses the IBM Telco Customer Churn dataset from Kaggle (supplied by IBM). It contains 7,043 customer records with demographic, service, and billing features and a binary churn label.[web:44][web:54][web:58]

(You will link to the Kaggle page where you download it.)

---

## 3. Pipeline stages

### 3.1 Ingestion & validation

Implement a script `src/data/make_dataset.py`:

- Read raw CSV(s) from `data/raw/` and write a cleaned version to `data/processed/`.  
- Validate:
  - Expected columns exist (e.g., `customerID`, `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `Churn`).[web:44][web:56]  
  - Types are correct (numeric vs categorical).  
  - No duplicate `customerID` values.  

Optionally, integrate a data validation framework (e.g., Great Expectations) to define expectations like “`tenure` ≥ 0” and “`MonthlyCharges` > 0”.[web:48][web:58]

### 3.2 Exploratory data analysis (EDA)

Use notebooks only for exploration, e.g., `notebooks/01_eda.ipynb`:

- Inspect class distribution: what percentage of customers have `Churn = Yes` vs `No` (usually imbalanced).[web:48][web:58]  
- Explore relationships:
  - Churn by contract type (month‑to‑month vs one‑year vs two‑year).[web:56][web:61]  
  - Churn vs tenure, monthly charges, presence of internet/phone services.[web:48][web:58][web:63]  
- Summarize insights in your README as business statements, e.g., “Month‑to‑month contracts with high monthly charges and short tenure show higher churn rates.”[web:44][web:48][web:63]

### 3.3 Feature engineering

Create `src/features/build_features.py`:

- Preprocessing:
  - Convert `TotalCharges` to numeric and handle blanks or missing values.  
  - Encode categorical variables (`Contract`, `InternetService`, `PaymentMethod`, etc.) using one‑hot or target encoding.[web:48][web:58]  
  - Scale numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) with StandardScaler or MinMaxScaler if needed.[web:54]  
- Feature ideas (derived from churn research):[web:48][web:55][web:63]  
  - Flags: `is_month_to_month`, `has_multiple_services`, `is_senior`, `has_fiber`.  
  - Ratios: `MonthlyCharges / (TotalCharges / tenure)` for stable billing patterns, where defined.  
- Split into train/validation/test with stratification on the churn label to preserve class balance.[web:48][web:58]

### 3.4 Model training pipeline

Implement `src/models/train_model.py`:

- Baseline models:
  - Logistic Regression and Random Forest, both commonly used on this dataset.[web:48][web:56][web:58]  
- Advanced models:
  - Gradient Boosting models (e.g., XGBoost, LightGBM, CatBoost) and optionally deep models like TabNet or FT‑Transformer if you want to go further.[web:44][web:55][web:61]  
- Handle class imbalance:
  - Use class weights or oversampling methods like SMOTE/ADASYN or hybrid methods (SMOTEENN, SMOTETomek) that have been evaluated on this dataset.[web:39][web:47][web:61][web:62]  
- Evaluation:
  - Metrics: accuracy, precision, recall, F1, ROC‑AUC, confusion matrix.[web:48][web:56][web:58]  
  - Compare algorithms, and keep a table in your README with metric scores.  

Add **experiment tracking** via MLflow:

- Log parameters (model type, hyperparameters), metrics, and model artifacts to an MLflow tracking server.  
- This mirrors how automated ML pipelines for tabular data are typically structured.[web:29][web:33][web:35]

### 3.5 Model explanation

Create `src/models/explain_model.py`:

- Use SHAP or permutation importance to identify top drivers of churn; prior work emphasizes factors such as contract type, monthly charges, and tenure.[web:55][web:58][web:63]  
- Generate:
  - Global importance plot (e.g., SHAP summary plot).  
  - Example‑level explanations for a few churn vs non‑churn customers.  
- Save plots to `reports/figures/` and reference them in the README.

### 3.6 Serving: prediction API

Implement `src/serving/app.py` with **FastAPI** (or Flask):

- On startup, load the trained model and preprocessing pipeline from `models/model.joblib` (and a transformer object).  
- Define a Pydantic model for request schema: it should mirror the customer features (contract, tenure, charges, etc.).  
- Provide endpoints:
  - `POST /predict` that accepts a single customer or list of customers and returns:
    - `churn_probability` (e.g., 0–1).  
    - `churn_label` (Yes/No based on a threshold).  
- This structure is similar in spirit to many research and industry deployments of churn models in telecom.[web:51][web:53]

**Dockerization:**

- Write a `Dockerfile` that:
  - Installs dependencies (Python base image, `pip install -r requirements.txt`).  
  - Exposes the port (e.g., 8000) and launches `uvicorn src.serving.app:app`.  
- Optionally provide a `docker-compose.yml` to run the app plus an MLflow tracking server or a database.

### 3.7 Batch inference & monitoring

Create `src/pipeline/batch_inference.py`:

- Read a CSV of new customers from `data/new_customers/`.  
- Apply preprocessing + model to generate predictions and write them to `data/predictions/`.  

Add simple monitoring:

- Compute:
  - Distribution of predicted churn probabilities over time.  
  - Basic data drift checks (e.g., compare feature distributions of new data vs training data).[web:18][web:53]  
- If you simulate ground‑truth labels later, periodically recompute performance metrics on recent predictions.

---

## 4. Orchestration and automation

### 4.1 Workflow orchestration

Use Airflow or Prefect (or a custom Python orchestrator) to define your pipeline:

- Training DAG/flow:
  - `ingest_data → validate_data → build_features → train_model → evaluate_model → explain_model`.  
- Scoring DAG/flow:
  - `ingest_new_data → batch_inference → monitoring`.  

Each stage calls the corresponding script in `src/`.

### 4.2 CI/CD with GitHub Actions

Configure a simple CI workflow in `.github/workflows/ci.yml`:

- On each push or pull request:
  - Install dependencies.  
  - Run unit tests in `tests/`.  
  - Run code style checks (e.g., `black`, `flake8`, `isort`).  
- Optionally:
  - Build the Docker image.  
  - Run a minimal smoke test (e.g., import the FastAPI app and call a prediction with dummy data).

This aligns with recommendations for treating ML systems as full software products, not just models.[web:24][web:31]

---

## 5. Suggested repo structure

A professional‑looking layout might be:

```text
telco-churn-pipeline/
  README.md
  pyproject.toml or requirements.txt
  data/
    raw/             # (empty, gitignored; instructions in README)
    processed/
    predictions/
    new_customers/
  models/
  notebooks/
    01_eda.ipynb
    02_model_prototyping.ipynb
  src/
    data/
      make_dataset.py
    features/
      build_features.py
    models/
      train_model.py
      evaluate_model.py
      explain_model.py
    serving/
      app.py
    pipeline/
      batch_inference.py
      run_pipeline.py
  configs/
    config.yaml       # paths, model params, etc.
  tests/
    test_data.py
    test_features.py
    test_models.py
    test_api.py
  Dockerfile
  .github/
    workflows/
      ci.yml
