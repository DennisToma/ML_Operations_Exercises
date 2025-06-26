# Lending Club Loan Default Prediction Pipeline (Dockerized)

This project implements a machine learning pipeline to predict loan defaults using the Lending Club dataset. The pipeline is containerized using Docker and orchestrated with Docker Compose, breaking down the process into distinct services: data validation, model training, and model validation (robustness testing).

## Project Structure

lending_club_pipeline/  
├── data/  
│ └── accepted_2007_to_2018Q4.csv # Dataset  
├── mlruns/ # MLflow tracking data (auto-created)  
├── outputs/ # Inter-container communication & reports  
│ ├── data_quality_report.json # Output from data validation  
│ └── model_meta.json # Trained model URI, run ID, features  
├── Dockerfile.data_validator # Dockerfile for data validation service  
├── Dockerfile.model_trainer # Dockerfile for model training service  
├── Dockerfile.model_validator # Dockerfile for model validation service  
├── docker-compose.yml # Docker Compose orchestration file  
├── requirements-data-validator.txt # Python dependencies for data validation  
├── requirements-model-trainer.txt # Python dependencies for model training  
├── requirements-model-validator.txt # Python dependencies for model validation  
├── data_quality_tests.py # Script with data quality check logic  
├── run_data_validation.py # Entrypoint script for data validation container  
├── run_model_training.py # Entrypoint script for model training container  
├── run_model_validation.py # Entrypoint script for model validation container  
└── README.md # This file  

## Prerequisites

*   Docker: [Install Docker](https://docs.docker.com/get-docker/)
*   Docker Compose: Usually included with Docker Desktop. If not, [Install Docker Compose](https://docs.docker.com/compose/install/)

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd lending_club_pipeline
    ```
2.  **Place Data:**
    Ensure your dataset (e.g., `accepted_2007_to_2018Q4.csv`) is placed inside the `data/` directory.
3.  **Review Configuration (Optional):**
    You can adjust environment variables within `docker-compose.yml` for each service, such as:
    *   `DATA_PATH`: Path to the dataset.
    *   `MLFLOW_EXPERIMENT_NAME`: Name for the MLflow experiment.
    *   `USE_SAMPLE` & `SAMPLE_SIZE` in `model-trainer`: To use a subset of data for faster runs during development.
    *   `MODEL_VERSION_NAME` & `REGULARIZATION_C` in `model-trainer`: To control model versioning and hyperparameters.

## Running the Pipeline

The project now supports multiple, versioned flows.

### 1. Full Workflow Walkthrough

This section provides a complete, end-to-end walkthrough to demonstrate all features of the project. Run these commands from the project's root directory in sequence.

**Step 1: Train and Validate the V1 Model**

This is the baseline pipeline. It runs data quality checks, trains the default model, and performs robustness validation.

```bash
docker-compose up --build
```
*   **Outcome:** The V1 model ("Model_V1_Default") will be trained and logged in MLflow. A metadata file `outputs/model_meta_Model_V1_Default.json` will be created containing its Run ID. Data quality, validation, and robustness reports will also be generated in `outputs/`.

**Step 2: Run the Post-Deployment Drift Monitor**

This simulates a check for data drift after the V1 model has been "deployed". It compares the training data statistics (captured during the previous run) with the full, "incoming" production dataset.

```bash
docker-compose -f docker-compose.yml run --rm drift-monitor
```
*   **Outcome:** A drift report `outputs/drift_report.json` will be generated. As expected, it will show significant drift, proving the monitoring system works correctly.

**Step 3: Train and Validate the V2 Model**

This command uses a different configuration to train a second version of the model with stronger regularization.

```bash
docker-compose -f docker-compose-v2.yml up --build
```
*   **Outcome:** The V2 model ("Model_V2_Strong_Reg") will be trained and logged in MLflow. A second metadata file `outputs/model_meta_Model_V2_Strong_Reg.json` will be created.

**Step 4: Run the A/B Test**

This is the final step, where the two model versions are compared head-to-head on unseen data. You must first get the Run IDs for both models from the JSON files created in steps 1 and 3.

1.  Open `outputs/model_meta_Model_V1_Default.json` and copy the `mlflow_run_id`.
2.  Open `outputs/model_meta_Model_V2_Strong_Reg.json` and copy the `mlflow_run_id`.
3.  Set these as environment variables and run the A/B test compose file.

```bash
# Example for PowerShell:
$env:MODEL_RUN_ID_1="<ID_from_V1_file>"
$env:MODEL_RUN_ID_2="<ID_from_V2_file>"
docker-compose -f docker-compose-ab-test.yml up --build

# Example for Bash (Linux/macOS):
# export MODEL_RUN_ID_1="<ID_from_V1_file>"
# export MODEL_RUN_ID_2="<ID_from_V2_file>"
# docker-compose -f docker-compose-ab-test.yml up --build
```
*   **Outcome:** A final report, `outputs/ab_test_report_v1_vs_v2_comparison.json`, is created, showing the comparative ROC AUC scores for both models and allowing for a data-driven decision.

### 2. Running the V1 Training Pipeline

This is the main, multi-step pipeline that performs the core tasks of data validation, model training, and model validation.

### Step 1: Run the V1 Training Flow
This runs the default pipeline with standard hyperparameters.

```bash
docker-compose up
```
This command will execute the `data-validator`, `model-trainer`, `model-validator`, and `drift-monitor` services in sequence. After it completes, **find the MLflow Run ID** for the `Model_V1_Default` run from the console output or by running `mlflow ui`.

### Step 2: Run the V2 Training Flow
This runs a second version of the pipeline with different hyperparameters (stronger regularization) to create a different model.

```bash
docker-compose -f docker-compose-v2.yml up
```
Again, after this completes, **find the MLflow Run ID** for the `Model_V2_Strong_Reg` run.

### Step 3: Configure and Run the A/B Test
This flow compares the performance of the two models you just trained.

1.  **Configure:** Open the `docker-compose-ab-test.yml` file. Find the `MLFLOW_RUN_IDS` environment variable.
2.  **Edit:** Replace `PASTE_V1_RUN_ID_HERE` and `PASTE_V2_RUN_ID_HERE` with the actual Run IDs you noted in the previous steps.
    ```yaml
    # ...
    environment:
      # ...
      - MLFLOW_RUN_IDS=a1b2c3d4e5f6,1a2b3c4d5e6f # <-- Paste your Run IDs here
    ```
3.  **Run:** Execute the A/B test flow.
    ```bash
    docker-compose -f docker-compose-ab-test.yml up
    ```

### Stopping and Cleanup
```bash
# Stop the currently running flow
docker-compose down

# Stop the V2 flow if it was the last one run
docker-compose -f docker-compose-v2.yml down
```

## Services

### 1. Data Validator (`data-validator`)
*   **Script:** `run_data_validation.py` (uses `data_quality_tests.py`)
*   **Purpose:** Performs initial data quality checks on the dataset.
*   **Output:**
    *   A `data_quality_report.json` file in the `outputs/` directory.
    *   The pipeline will halt if critical data quality checks fail.

### 2. Model Trainer (`model-trainer`)
*   **Script:** `run_model_training.py`
*   **Purpose:**
    *   Loads and preprocesses the data.
    *   Trains a Logistic Regression model to predict loan default.
    *   Logs the model, parameters, metrics (accuracy, ROC AUC), and a classification report to MLflow.
    *   Registers the trained model in MLflow.
*   **Output:**
    *   MLflow artifacts (model, metrics, params) stored in the `mlruns/` directory.
    *   A versioned `model_meta_<VERSION_NAME>.json` file in `outputs/` for the next step.

### 3. Model Validator (`model-validator`)
*   **Script:** `run_model_validation.py`
*   **Purpose:**
    *   Loads the trained model and test data.
    *   Performs robustness tests by adding noise to key features and observing changes in predictions.
    *   Logs robustness metrics back to the original MLflow run associated with the trained model.
*   **Output:**
    *   Additional metrics and parameters logged to the existing MLflow run in `mlruns/`.

### 4. Drift Monitor (`drift-monitor`)
*   **Script:** `run_drift_monitoring.py`
*   **Purpose:**
    *   Runs after the main pipeline completes.
    *   Performs covariate drift detection by comparing the first 10% of the data (reference) against the last 10% (current).
    *   Uses the two-sample Kolmogorov-Smirnov (K-S) test to check for significant changes in feature distributions.
*   **Output:**
    *   A `drift_report.json` file in the `outputs/` directory.

### 5. A/B Tester (`ab-tester`)
*   **Script:** `run_ab_test.py`
*   **Purpose:**
    *   Loads two different model versions specified by their MLflow Run IDs.
    *   Deterministically splits a test dataset between the models.
    *   Evaluates and compares their performance (e.g., ROC AUC).
*   **Output:**
    *   A versioned `ab_test_report_<TEST_ID>.json` file in `outputs/` with the comparative results.

### 7. Interpreting the A/B Test Results

After running the A/B test, a JSON report is generated in the `outputs` folder. This report is the key outcome of Task 3, providing a direct comparison between your two model versions.

**Example Report (`outputs/ab_test_report_v1_vs_v2_comparison.json`):**
```json
{
  "test_id": "v1_vs_v2_comparison",
  "models": {
    "Model_0": {
      "run_id": "2d16f5bde30c4ecb82bff2710a52a9cb",
      "roc_auc": 0.6773,
      "test_data_size": 80279
    },
    "Model_1": {
      "run_id": "0c112bec5f064f4bb9f0559b286050f9",
      "roc_auc": 0.6698,
      "test_data_size": 80422
    }
  }
}
```

**How to Interpret This:**

1.  **Metric (ROC AUC):** The `roc_auc` score measures the model's ability to distinguish between positive and negative classes (loan defaults vs. fully paid loans). A higher score is better.
2.  **Comparison:** In this example, "Model_0" (the V1 model with default settings) achieved a score of `0.6773`, while "Model_1" (the V2 model with stronger regularization) scored `0.6698`.
3.  **Conclusion:** The V1 model performed slightly better on the unseen test data. This is a successful and valuable result. It demonstrates that the hyperparameter tuning attempted in V2 was not beneficial, and the organization should stick with the V1 model for deployment.

The A/B test provides the empirical evidence needed to make a data-driven decision about which model version to use, which is a core practice in MLOps.

## Key Findings from Pipeline Runs

### Data Drift Detection
When the pipeline is run, the `drift-monitor` service performs a covariate drift check on key features, comparing the earliest data in the set (the "reference" set) to the latest data (the "current" set).

**Result:** The monitor consistently and correctly reports a **`"DRIFT_DETECTED"`** status.

The output report (`outputs/drift_report.json`) shows that for all monitored features (`loan_amnt`, `int_rate`, `annual_inc`, etc.), the p-value from the Kolmogorov-Smirnov test is effectively zero. This provides strong statistical evidence that the distribution of these features has significantly changed over the timespan of the dataset (2007-2018).

This is an **excellent and expected result**. It demonstrates that the monitoring component is working as intended. The vast economic changes between 2007 and 2018 (e.g., the 2008 financial crisis, subsequent recovery, and changing interest rate environments) mean that the data is not stationary. A model trained on early data would likely perform poorly on recent data. The drift detector successfully identifies this critical, real-world issue, highlighting the necessity of continuous monitoring and model retraining in a production environment.

## Accessing MLflow UI

Since MLflow is configured to use a local file system (`file:///app/mlruns` inside containers, mapped to `./mlruns` on your host), you can view the MLflow UI by running the MLflow server locally, pointing it to the `mlruns` directory:

1.  Ensure you have MLflow installed in your local Python environment (outside Docker):
    ```bash
    pip install mlflow
    ```
2.  Navigate to your project's root directory (`lending_club_pipeline/`) in your terminal.
3.  Run the MLflow UI:
    ```bash
    mlflow ui
    ```
    This will typically start the server on `http://localhost:5000`. Open this URL in your browser to see your experiments, runs, and artifacts.

## Outputs Summary

*   **Data Quality Report:** `outputs/data_quality_report.json`
*   **Drift Report:** `outputs/drift_report.json`
*   **A/B Test Report:** `outputs/ab_test_report_v1_vs_v2_comparison.json`
*   **Model Metadata:** `outputs/model_meta_*.json` (links to the trained models in MLflow)
*   **MLflow Runs & Artifacts:** `./mlruns/` (viewable with `mlflow ui`)
