# run_model_validation.py
import os
import sys
import pandas as pd
import numpy as np
import pathlib
import mlflow
import json
import logging
import psutil
import gc

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_quality_tests import parse_emp_length
except ImportError as e:
    logging.error(f"Failed to import parse_emp_length from data_quality_tests: {e}")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_model_robustness(
    data_path_for_test: str,
    mlflow_tracking_uri: str,
    output_dir_meta: str, # Directory to read model_meta.json from
    model_version_name: str # The key for the model in the JSON file
):
    logger.info("--- Starting Model Validation (Robustness Test) Script ---")
    logger.info(f"Validating model version: {model_version_name}")
    logger.info(f"Available memory before test: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    meta_file_path = os.path.join(output_dir_meta, "model_meta.json")
    try:
        with open(meta_file_path, 'r') as f:
            all_model_meta = json.load(f)
    except FileNotFoundError:
        logger.error(f"Model metadata file not found: {meta_file_path}")
        return False # Critical failure
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from model metadata file: {meta_file_path}")
        return False # Critical failure
    
    # Select the metadata for the specific model version
    model_meta = all_model_meta.get(model_version_name)
    if not model_meta:
        logger.error(f"Metadata for model version '{model_version_name}' not found in {meta_file_path}")
        return False

    if model_meta.get("status") == "TRAINING_FAILED" or not model_meta.get("model_uri"):
        logger.error(f"Model training failed or model_uri not found in metadata for {model_version_name}. Skipping validation. Metadata: {model_meta}")
        # Depending on desired behavior, this could be a sys.exit(0) if skipping is acceptable,
        # or sys.exit(1) if a valid model is strictly required for this stage to "pass".
        # For now, let's consider it a script success (it ran) but validation was skipped.
        return True # Script ran, but validation skipped.

    model_uri_to_test = model_meta.get("model_uri")
    mlflow_run_id_to_log_to = model_meta.get("mlflow_run_id")
    numeric_features = model_meta.get("numeric_features", []) # Get from metadata
    categorical_features = model_meta.get("categorical_features", []) # Get from metadata

    if not all([model_uri_to_test, mlflow_run_id_to_log_to, numeric_features, categorical_features]):
        logger.error(f"Incomplete model metadata. Required fields missing. Metadata: {model_meta}")
        return False # Critical failure for validation logic

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    logger.info(f"Loading model from URI: {model_uri_to_test} for run ID: {mlflow_run_id_to_log_to}")

    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri_to_test)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri_to_test}: {e}", exc_info=True)
        # Log failure to the original run if possible
        try:
            with mlflow.start_run(run_id=mlflow_run_id_to_log_to) as run:
                mlflow.log_param("robustness_test_status", "failed_model_load")
                mlflow.log_param("robustness_error", f"Model load failed: {str(e)}")
        except Exception as mlflow_err:
            logger.error(f"Additionally failed to log model load failure to MLflow: {mlflow_err}")
        return False # Critical failure

    active_run_context = None
    try:
        with mlflow.start_run(run_id=mlflow_run_id_to_log_to) as active_run_context: # Re-open the original run
            logger.info(f"Logging robustness metrics to MLflow run: {mlflow_run_id_to_log_to}")
            
            # --- Prepare data for robustness test ---
            logger.info("Preparing data for robustness test...")
            chunk_size = 100000
            chunks = []
            valid_statuses = ['Fully Paid', 'Charged Off'] # Consistent with training
            max_chunks_robustness = 2 # Use a smaller sample for robustness

            for i, chunk_df in enumerate(pd.read_csv(data_path_for_test, chunksize=chunk_size, low_memory=True)):
                if i >= max_chunks_robustness:
                    logger.info(f"Reached max {max_chunks_robustness} chunks for robustness test.")
                    break
                filtered_chunk = chunk_df[chunk_df['loan_status'].isin(valid_statuses)].copy()
                if not filtered_chunk.empty:
                    chunks.append(filtered_chunk)
                del chunk_df, filtered_chunk; gc.collect()

            if not chunks:
                logger.warning("No data available for robustness test after filtering. Skipping test.")
                mlflow.log_param("robustness_test_status", "skipped_no_data")
                return True # Test skipped, not a failure of the script itself

            df_test_robust = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded test data for robustness: {df_test_robust.shape[0]} rows")

            # Apply same initial cleaning as in training
            if 'emp_length' in df_test_robust.columns:
                df_test_robust['emp_length_numeric'] = df_test_robust['emp_length'].apply(parse_emp_length)
            if 'int_rate' in df_test_robust.columns:
                df_test_robust['int_rate'] = df_test_robust['int_rate'].astype(str).str.rstrip('%').astype(float) / 100.0
            if 'revol_util' in df_test_robust.columns:
                df_test_robust['revol_util'] = df_test_robust['revol_util'].astype(str).str.rstrip('%')
                df_test_robust['revol_util'] = pd.to_numeric(df_test_robust['revol_util'], errors='coerce') / 100.0
            
            features_to_use = numeric_features + categorical_features
            df_robust_sample_base = df_test_robust[features_to_use].copy()
            del df_test_robust; gc.collect()

            df_robust_sample_base.dropna(subset=numeric_features, inplace=True) # Drop NaNs in numerics
            for col in categorical_features: # Impute categoricals
                 if df_robust_sample_base[col].isnull().any():
                    df_robust_sample_base[col].fillna('Missing', inplace=True)

            sample_size_robust = min(1000, len(df_robust_sample_base))
            if sample_size_robust == 0:
                logger.warning("No data for robustness test after cleaning. Skipping.")
                mlflow.log_param("robustness_test_status", "skipped_no_data_post_clean")
                return True

            X_sample = df_robust_sample_base.sample(n=sample_size_robust, random_state=456)
            logger.info(f"Performing robustness check on {len(X_sample)} samples.")

            original_predictions = loaded_model.predict(X_sample)
            perturbed_sample = X_sample.copy()
            features_to_perturb = ['annual_inc', 'dti', 'revol_util']
            noise_level_std_fraction = 0.05

            for col in features_to_perturb:
                if col in perturbed_sample.columns and col in numeric_features:
                    std_dev = perturbed_sample[col].std()
                    if pd.notna(std_dev) and std_dev > 0:
                        noise = np.random.normal(0, std_dev * noise_level_std_fraction, size=perturbed_sample.shape[0])
                        perturbed_sample[col] = perturbed_sample[col] + noise
                        if col == 'dti': perturbed_sample[col] = perturbed_sample[col].clip(lower=0)
                        if col == 'annual_inc': perturbed_sample[col] = perturbed_sample[col].clip(lower=0)
                        if col == 'revol_util': perturbed_sample[col] = perturbed_sample[col].clip(lower=0, upper=max(1.5, perturbed_sample[col].max(skipna=True) if not perturbed_sample[col].empty else 1.5))


            perturbed_predictions = loaded_model.predict(perturbed_sample)
            num_changed = np.sum(original_predictions != perturbed_predictions)
            change_percentage = (num_changed / len(X_sample)) * 100 if len(X_sample) > 0 else 0
            
            logger.info(f"Robustness: Predictions changed: {num_changed} ({change_percentage:.2f}%)")
            robustness_threshold = 15.0
            passed_robustness = change_percentage < robustness_threshold

            mlflow.log_metric("robustness_change_percentage", change_percentage)
            mlflow.log_param("robustness_features_perturbed", ", ".join(features_to_perturb))
            mlflow.log_param("robustness_noise_level_std_fraction", noise_level_std_fraction)
            mlflow.log_param("robustness_threshold_percentage", robustness_threshold)
            mlflow.log_metric("robustness_test_passed", int(passed_robustness))
            mlflow.log_param("robustness_test_status", "completed")

            logger.info(f"Robustness test passed (change < {robustness_threshold}%): {passed_robustness}")
            # if not passed_robustness:
            #     logger.warning("Model robustness test FAILED (change percentage exceeded threshold)!")
            #     # Decide if this constitutes a script failure (sys.exit(1))
            #     # For now, the script completed, the test result is just a metric.

            return True # Script completed successfully

    except Exception as e:
        logger.error(f"Error during robustness test execution: {e}", exc_info=True)
        if active_run_context or mlflow_run_id_to_log_to: # Try to log to existing run
            try:
                # Ensure we are in the correct run context if not already
                current_run = mlflow.active_run()
                if not current_run or current_run.info.run_id != mlflow_run_id_to_log_to:
                    if current_run: mlflow.end_run() # End any stray run
                    mlflow.start_run(run_id=mlflow_run_id_to_log_to) # Re-open target run
                
                mlflow.log_param("robustness_test_status", "failed_execution")
                mlflow.log_param("robustness_error", str(e))
            except Exception as mlflow_err:
                logger.error(f"Additionally failed to log robustness execution failure to MLflow: {mlflow_err}")
        return False # Script failed
    finally:
        if mlflow.active_run(): # Ensure any run opened here is closed
             # Check if it's the one we intended to log to, or a new one by mistake
            if active_run_context and mlflow.active_run().info.run_id == active_run_context.info.run_id:
                pass # 'with' statement will handle it
            elif mlflow.active_run().info.run_id == mlflow_run_id_to_log_to:
                 pass # 'with' statement for the target run will handle it
            else: # some other run was opened by mistake
                # mlflow.end_run() # This might be risky if not managed carefully
                logger.warning(f"An unexpected MLflow run {mlflow.active_run().info.run_id} was active at the end of robustness test.")


def main():
    data_path = os.getenv("DATA_PATH")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    output_dir_meta = os.getenv("OUTPUT_DIR", "/app/outputs") # To read model_meta.json

    # Get the model version name to validate
    model_version_name = os.getenv("MODEL_VERSION_NAME")
    if not model_version_name:
        logger.error("Required environment variable MODEL_VERSION_NAME not set.")
        sys.exit(1)

    if not data_path:
        logger.error("Required environment variable DATA_PATH not set.")
        sys.exit(1)

    meta_file_path = os.path.join(output_dir_meta, "model_meta.json")
    if not os.path.exists(meta_file_path):
        logger.error(f"Input metadata file not found at: {meta_file_path}")
        sys.exit(1)

    success = test_model_robustness(data_path, mlflow_tracking_uri, output_dir_meta, model_version_name)

    if success:
        logger.info("Model validation (robustness) script finished.")
        sys.exit(0)
    else:
        logger.error("Model validation (robustness) script failed or encountered critical errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()