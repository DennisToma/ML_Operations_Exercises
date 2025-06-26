# Lending Club Loan Default Prediction Pipeline (Dockerized)

This project implements a machine learning pipeline to predict loan defaults using the Lending Club dataset. The pipeline is containerized using Docker and orchestrated with Docker Compose, breaking down the process into distinct, version-controlled services.

The key improvement in this version is a **dynamic model versioning system**. Instead of manually passing MLflow Run IDs between steps, the pipeline now uses a central `outputs/model_meta.json` file. The training service populates this file, and downstream services (validation, A/B testing) read from it using logical model names (e.g., `Model_V1_Default`, `Model_V2_Strong_Reg`), making the entire workflow more robust, automated, and less error-prone.

Additionally, the model training script has been **optimized for memory efficiency**. It now selectively loads only the required data columns, significantly reducing its memory footprint and preventing `exit code 137` errors on large datasets.

## Project Structure

```
lending_club_pipeline/
├── data/
│   └── accepted_2007_to_2018Q4.csv       # Dataset
├── mlruns/                                 # MLflow tracking data (auto-created)
├── outputs/                                # Inter-container communication & reports
│   ├── data_quality_report.json          # Output from data validation
│   └── model_meta.json                   # Central metadata for all trained models
├── Dockerfile.*                            # Dockerfiles for each service
├── docker-compose.yml                      # Main pipeline for V1 model
├── docker-compose-v2.yml                   # Pipeline for V2 model
├── docker-compose-ab-test.yml              # Pipeline for A/B testing
├── requirements-*.txt                      # Python dependencies for each service
├── run_*.py                                # Entrypoint scripts for services
└── README.md                               # This file
```

## Prerequisites

*   Docker: [Install Docker](https://docs.docker.com/get-docker/)
*   Docker Compose: Usually included with Docker Desktop.

## Setup

1.  **Clone the repository.**
2.  **Place Data:** Ensure your dataset (e.g., `accepted_2007_to_2018Q4.csv`) is in the `data/` directory.
3.  **Review Configuration (Optional):**
    You can adjust environment variables within the `docker-compose-*.yml` files:
    *   `MODEL_VERSION_NAME`: A logical name for a model being trained or validated (e.g., `Model_V1_Default`). This is used as the key in `model_meta.json`.
    *   `MODEL_VERSION_NAMES`: A comma-separated list of logical names for the A/B testing service to compare (e.g., `Model_V1_Default,Model_V2_Strong_Reg`).
    *   `USE_SAMPLE` & `SAMPLE_SIZE`: Use a subset of data for faster development runs.

## Running the End-to-End Pipeline

This workflow demonstrates how to train two different model versions and then run an A/B test to compare them. The process is now fully automated, with no need to manually find and copy MLflow Run IDs.

### Step 1: Train and Validate the V1 Model

This command runs the baseline pipeline. It performs data quality checks, trains the "V1" model, and validates its robustness.

```bash
docker-compose up --build
```
*   **Outcome:** The V1 model (`Model_V1_Default`) is trained and logged to MLflow. Its metadata is automatically saved to `outputs/model_meta.json`.

### Step 2: Train and Validate the V2 Model

This command uses a different configuration (`docker-compose-v2.yml`) to train a second model version with stronger regularization.

```bash
docker-compose -f docker-compose-v2.yml up --build
```
*   **Outcome:** The V2 model (`Model_V2_Strong_Reg`) is trained and logged to MLflow. The `outputs/model_meta.json` file is automatically updated with this new model's metadata.

### Step 3: Run the A/B Test

This final step compares the two model versions on unseen data. It reads the model information directly from `model_meta.json` using the names specified in `docker-compose-ab-test.yml`.

```bash
docker-compose -f docker-compose-ab-test.yml up --build
```
*   **Outcome:** A final report, `outputs/ab_test_report_v1_vs_v2_comparison.json`, is created, showing the comparative performance (ROC AUC) of the V1 and V2 models.

### Stopping and Cleanup
```bash
# Stop and remove containers from the last command
docker-compose -f docker-compose-ab-test.yml down

# General cleanup for all flows
docker-compose down && docker-compose -f docker-compose-v2.yml down
```

## Services

### 1. Data Validator (`data-validator`)
*   **Script:** `run_data_validation.py`
*   **Purpose:** Performs initial data quality checks. Halts the pipeline on critical failures.
*   **Output:** `outputs/data_quality_report.json`.

### 2. Model Trainer (`model-trainer`)
*   **Script:** `run_model_training.py`
*   **Purpose:**
    *   Trains a model based on the `MODEL_VERSION_NAME` and other hyperparameters.
    *   Logs the model, parameters, and metrics to MLflow.
    *   **Crucially, it saves the model's metadata (MLflow URI, run ID, features) to the central `outputs/model_meta.json` file under its version name.**
*   **Key Improvement:** Uses a memory-optimized data loading strategy (`usecols`) to handle large datasets efficiently.

### 3. Model Validator (`model-validator`)
*   **Script:** `run_model_validation.py`
*   **Purpose:**
    *   **Reads `outputs/model_meta.json` to find the model URI corresponding to the `MODEL_VERSION_NAME` set in its environment.**
    *   Performs robustness tests by adding noise to features.
    *   Logs robustness metrics back to the original MLflow run.

### 4. Drift Monitor (`drift-monitor`)
*   **Script:** `run_drift_monitoring.py`
*   **Purpose:** Performs covariate drift detection by comparing training and production data slices.
*   **Output:** `outputs/drift_report.json`.

### 5. A/B Tester (`ab-tester`)
*   **Script:** `run_ab_test.py`
*   **Purpose:**
    *   **Reads `outputs/model_meta.json` to find all models specified in the `MODEL_VERSION_NAMES` environment variable.**
    *   Loads the models, splits test data between them, and evaluates performance.
*   **Output:** An A/B test report in `outputs/` comparing the models' ROC AUC scores.

## Accessing MLflow UI

You can view the MLflow UI by running the server locally, pointing it to the `mlruns` directory created by the pipeline.

1.  Install MLflow locally: `pip install mlflow`
2.  From the project's root directory, run:
    ```bash
    mlflow ui
    ```
This will start a local server (usually at `http://127.0.0.1:5000`) where you can browse all experiments, runs, and logged artifacts.
