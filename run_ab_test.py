import os
import sys
import json
import logging
import pathlib
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import roc_auc_score
import hashlib
import collections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ab_test(
    data_path: str,
    output_file_path: str,
    mlflow_tracking_uri: str,
    model_version_names: list,
    meta_file_path: str,
    test_id: str
):
    """
    Performs an A/B test between two or more model versions.

    This script loads different model versions specified by their MLflow Run IDs,
    divides a test dataset among them, evaluates their performance, and logs a
    comparative report.

    Data Splitting:
    The test data is split randomly but deterministically using a hash of the
    loan 'id' column. This ensures that a specific loan is always assigned to the
    same model version, which is crucial for reproducible A/B tests.

    Managing Multiple Tests:
    To manage multiple concurrent or subsequent A/B tests, this script uses a
    unique `test_id`. This ID is used to name the output report and can be
    used as a tag in MLflow to group all artifacts and results related to a
    single A/B test campaign. For example, one could have `test_id='q4_promo_model_test'`
    and another `test_id='new_features_beta_test'`, allowing for clear separation
    and analysis of results from different experiments.

    Args:
        data_path (str): Path to the unseen test data CSV.
        output_file_path (str): Path to save the JSON A/B test report.
        mlflow_tracking_uri (str): The MLflow tracking URI.
        model_version_names (list): A list of model version names (keys in metadata file).
        meta_file_path (str): Path to the model_meta.json file.
        test_id (str): A unique identifier for this specific A/B test.
    """
    logger.info(f"--- Starting A/B Test: {test_id} ---")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    report = {"test_id": test_id, "models": {}}
    
    try:
        # Load the model metadata
        logger.info(f"Loading model metadata from {meta_file_path}")
        with open(meta_file_path, 'r') as f:
            all_model_meta = json.load(f)

        # Memory-efficiently load the "unseen" data (last part of the file)
        logger.info(f"Memory-efficiently loading unseen data from {data_path}")
        
        chunk_size = 50000
        # Use a fixed number of chunks to represent the unseen test data
        num_chunks_for_set = 4 # e.g., last 200k rows
        
        current_chunks_deque = collections.deque(maxlen=num_chunks_for_set)
        iterator = pd.read_csv(data_path, chunksize=chunk_size, low_memory=False)
        
        for chunk in iterator:
            current_chunks_deque.append(chunk)
            
        unseen_df = pd.concat(list(current_chunks_deque), ignore_index=True)
        logger.info(f"Test data size: {len(unseen_df)} rows")

        # Define target and clean data
        unseen_df['target'] = unseen_df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
        
        # Simple preprocessing for consistency
        if 'emp_length' in unseen_df.columns:
            from data_quality_tests import parse_emp_length
            unseen_df['emp_length_numeric'] = unseen_df['emp_length'].apply(parse_emp_length)
        if 'int_rate' in unseen_df.columns:
            unseen_df['int_rate'] = unseen_df['int_rate'].astype(str).str.rstrip('%').astype(float) / 100.0
        if 'revol_util' in unseen_df.columns:
            unseen_df['revol_util'] = pd.to_numeric(unseen_df['revol_util'].astype(str).str.rstrip('%'), errors='coerce') / 100.0

        # Split data deterministically among models
        num_models = len(model_version_names)
        unseen_df['model_group'] = unseen_df['id'].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % num_models
        )
        logger.info(f"Data split into {num_models} groups for A/B testing.")

        for i, model_name in enumerate(model_version_names):
            logger.info(f"--- Evaluating Model Group {i} (Name: {model_name}) ---")
            
            model_meta = all_model_meta.get(model_name)
            if not model_meta:
                logger.error(f"Metadata for model '{model_name}' not found in metadata file. Skipping.")
                report["models"][model_name] = {"error": "Metadata not found"}
                continue

            # Get model URI and features from the metadata
            model_uri = model_meta.get("model_uri")
            run_id = model_meta.get("mlflow_run_id")
            numeric_features = model_meta.get("numeric_features", [])
            categorical_features = model_meta.get("categorical_features", [])
            features_to_use = numeric_features + categorical_features

            if not model_uri:
                logger.error(f"model_uri for '{model_name}' not found in metadata. Skipping.")
                report["models"][model_name] = {"error": "model_uri not found in metadata"}
                continue

            logger.info(f"Loading model: {model_name} from {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri) # Use pyfunc for general compatibility
            
            # Get data for this model's group
            group_df = unseen_df[unseen_df['model_group'] == i].copy()
            logger.info(f"Group {i} data size: {len(group_df)} rows")
            
            if group_df.empty:
                logger.warning(f"No data assigned to model group {i}. Skipping.")
                continue

            X_test = group_df[features_to_use]
            y_test = group_df['target']
            
            # Evaluate performance
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
                logger.info(f"ROC AUC for {model_name}: {roc_auc:.4f}")
                report["models"][model_name] = {
                    "run_id": run_id,
                    "roc_auc": roc_auc,
                    "test_data_size": len(group_df)
                }
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}", exc_info=True)
                report["models"][model_name] = {"error": str(e)}

    except Exception as e:
        logger.error(f"An error occurred during the A/B test: {e}", exc_info=True)
        report['error'] = str(e)

    finally:
        logger.info(f"Saving A/B test report to {output_file_path}")
        pathlib.Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    data_path = os.getenv("DATA_PATH")
    output_dir = os.getenv("OUTPUT_DIR", "/app/outputs")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_version_names_str = os.getenv("MODEL_VERSION_NAMES")
    test_id = os.getenv("AB_TEST_ID", "default_ab_test")

    if not all([data_path, mlflow_tracking_uri, model_version_names_str]):
        logger.error("Missing required env vars: DATA_PATH, MLFLOW_TRACKING_URI, or MODEL_VERSION_NAMES")
        sys.exit(1)
        
    model_version_names = [name.strip() for name in model_version_names_str.split(',')]
    meta_file_path = os.path.join(output_dir, "model_meta.json")
    report_file_path = os.path.join(output_dir, f"ab_test_report_{test_id}.json")
    
    if not os.path.exists(meta_file_path):
        logger.error(f"Metadata file not found at {meta_file_path}")
        sys.exit(1)

    run_ab_test(data_path, report_file_path, mlflow_tracking_uri, model_version_names, meta_file_path, test_id)
    logger.info("A/B Test script finished.")

if __name__ == "__main__":
    main() 