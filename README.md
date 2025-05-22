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

## Running the Pipeline

1.  **Build the Docker images:**
    ```bash
    docker-compose build
    ```
2.  **Run the pipeline:**
    ```bash
    docker-compose up
    ```
    This will start the services in the defined order:
    1.  `data-validator`
    2.  `model-trainer` (depends on successful completion of `data-validator`)
    3.  `model-validator` (depends on successful completion of `model-trainer`)

    To run in detached mode (in the background):
    ```bash
    docker-compose up -d
    ```

3.  **Stopping the pipeline:**
    ```bash
    docker-compose down
    ```
    To stop and remove volumes (be cautious, this deletes `mlruns` and `outputs` if they are Docker-managed volumes, but here they are host-mounted so local data persists):
    ```bash
    docker-compose down -v
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
    *   A `model_meta.json` file in `outputs/` containing the MLflow run ID, model URI, and feature lists for the next step.

### 3. Model Validator (`model-validator`)
*   **Script:** `run_model_validation.py`
*   **Purpose:**
    *   Loads the trained model and test data.
    *   Performs robustness tests by adding noise to key features and observing changes in predictions.
    *   Logs robustness metrics back to the original MLflow run associated with the trained model.
*   **Output:**
    *   Additional metrics and parameters logged to the existing MLflow run in `mlruns/`.

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
*   **Model Metadata:** `outputs/model_meta.json` (links to the trained model in MLflow)
*   **MLflow Runs & Artifacts:** `./mlruns/` (viewable with `mlflow ui`)
