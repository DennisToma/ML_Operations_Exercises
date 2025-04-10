# pipeline.py
import os
import sys
import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import mlflow
from mlflow.models import infer_signature
import prefect
from prefect import flow, task, get_run_logger

try:
    # Ensure data_quality_tests.py is in the same directory or Python path
    from data_quality_tests import run_data_quality_checks, parse_emp_length
except ImportError:
     print("Error: Could not import from data_quality_tests.py")
     print("Ensure data_quality_tests.py exists and contains the necessary functions.")
     exit(1)

# --- MLflow Configuration ---
mlruns_path = pathlib.Path(os.getcwd()).resolve() / "mlruns"
MLFLOW_TRACKING_URI = mlruns_path.as_uri()
MLFLOW_EXPERIMENT_NAME = "LendingClubDefaultPrediction_Prefect"
print(f"Setting MLflow Tracking URI to: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# --- Task Definitions ---

@task
def perform_data_quality_check_task(data_path: str):
    """
    Prefect task to run data quality checks using the external script.
    Raises an exception if checks fail.
    """
    logger = get_run_logger()
    logger.info(f"--- Task: Data Quality Check on {data_path} ---")

    passed = run_data_quality_checks(data_path)

    if not passed:
        logger.error("Data quality checks failed!")
        raise ValueError("Data quality checks failed! Stopping the pipeline.")
    else:
        logger.info("Data quality checks passed.")
        return True

@task
def train_and_version_model_task(data_path: str, min_records: int, test_set_prop: float):
    """
    Prefect task to preprocess data, train a model, handle errors,
    serialize, and version with MLflow.
    Returns:
        tuple: (model_uri, mlflow_run_id, numeric_features, categorical_features) on success,
               raises Exception on failure.
    """
    logger = get_run_logger()
    logger.info("--- Task: Train Model ---")
    model_uri = None
    mlflow_run_id = None
    numeric_features = [] # Initialize to return even on failure before definition
    categorical_features = []

    # --- Start MLflow Run ---
    flow_run_context = prefect.context.get_run_context()

    if flow_run_context and hasattr(flow_run_context, 'task_run') and flow_run_context.task_run:
        flow_run_id = str(flow_run_context.task_run.flow_run_id) # Access flow_run_id via task_run
    else:
        # Fallback if context is somehow unavailable (e.g., running task standalone)
        flow_run_id = "local_run_no_context"

    with mlflow.start_run(run_name=f"training_prefect_{flow_run_id}") as run:
        mlflow_run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {mlflow_run_id}")
        mlflow.log_param("orchestrator", "prefect")
        mlflow.log_param("prefect_flow_run_id", str(flow_run_id))
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("min_training_records_threshold", min_records)
        mlflow.log_param("test_set_size", test_set_prop)

        try:
            # --- Load Data ---
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path, low_memory=False)
            logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

            # --- Preprocessing, Filtering, Target Definition ---
            valid_statuses = ['Fully Paid', 'Charged Off']
            df = df[df['loan_status'].isin(valid_statuses)].copy()
            logger.info(f"Filtered data to statuses {valid_statuses}: {df.shape[0]} rows remain.")
            df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

            # Apply cleaning functions
            if 'emp_length' in df.columns:
                df['emp_length_numeric'] = df['emp_length'].apply(parse_emp_length)
            if 'int_rate' in df.columns:
                df['int_rate'] = df['int_rate'].astype(str).str.rstrip('%').astype(float) / 100.0
            if 'revol_util' in df.columns:
                # Handle potential non-string entries before .str
                df['revol_util'] = df['revol_util'].astype(str).str.rstrip('%')
                df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce') / 100.0 # Coerce errors


            # --- Error Handling: Induced Error (Data Size) ---
            logger.info(f"Checking filtered data size: {len(df)} records.")
            if len(df) < min_records:
                error_message = f"Filtered data size ({len(df)}) is less than required minimum ({min_records})."
                logger.error(error_message)
                mlflow.log_param("training_status", "failed_insufficient_data")
                mlflow.log_param("error_message", error_message)
                raise ValueError(error_message)

            # --- Feature Selection ---
            # Define features available at loan issuance
            numeric_features = [
                'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
                'revol_bal', 'revol_util', 'total_acc'
            ]
            # Add parsed emp_length if it exists
            if 'emp_length_numeric' in df.columns:
                numeric_features.append('emp_length_numeric')

            categorical_features = [
                'term', 'grade', 'sub_grade', #'emp_length', # Using numeric version instead
                'home_ownership', 'verification_status', 'purpose',
                'addr_state' # High cardinality, careful consideration needed in real project
            ]
            # Ensure original emp_length is not in categorical if numeric is used
            if 'emp_length_numeric' in df.columns and 'emp_length' in categorical_features:
                 categorical_features.remove('emp_length')

            features_to_use = numeric_features + categorical_features
            mlflow.log_param("numeric_features_used", ", ".join(numeric_features))
            mlflow.log_param("categorical_features_used", ", ".join(categorical_features))

            # Drop rows with NaN in key numeric features or target BEFORE splitting
            # Also handle potential NaN in revol_util from coercion
            df.dropna(subset=numeric_features + ['target'], inplace=True)

            # Impute remaining NaNs in categoricals (if any after filtering/dropping)
            for col in categorical_features:
                if df[col].isnull().any():
                    df[col].fillna('Missing', inplace=True)

            logger.info(f"Data shape after cleaning/filtering/NaN handling: {df.shape}")

            X = df[features_to_use]
            y = df['target']

            # --- Split Data ---
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_set_prop, random_state=42, stratify=y
            )
            mlflow.log_param("train_set_size", len(X_train))
            mlflow.log_param("test_set_actual_size", len(X_test))

            # --- Preprocessing Pipeline ---
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Impute missing numeric values (just in case)
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), # Impute missing categoricals
                # Handle unknown categories gracefully during prediction, ignore features not seen in training
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop' # Drop columns not specified
            )

            # --- Model Definition ---
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', solver='liblinear'))]) # Added solver

            # --- Training ---
            logger.info("Training the model...")
            model.fit(X_train, y_train)
            logger.info("Model training complete.")

            # --- Evaluation ---
            logger.info("Evaluating the model...")
            y_pred_test = model.predict(X_test)
            # Handle potential error if model can't predict probabilities (e.g., some classifiers)
            try:
                y_proba_test = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba_test)
                mlflow.log_metric("test_roc_auc", roc_auc)
                logger.info(f"Test ROC AUC: {roc_auc:.4f}")
            except AttributeError:
                logger.warning("Model does not support predict_proba, skipping ROC AUC.")
                roc_auc = np.nan # Assign NaN if not calculable

            accuracy = accuracy_score(y_test, y_pred_test)
            report = classification_report(y_test, y_pred_test, target_names=['Fully Paid', 'Charged Off'], zero_division=0)
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"\nClassification Report:\n{report}")

            mlflow.log_metric("test_accuracy", accuracy)
            # Log ROC AUC only if calculated
            if not np.isnan(roc_auc):
                mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_text(report, "classification_report.txt")

            # --- Versioning ---
            logger.info("Logging model with MLflow...")
            input_sample = X_train.head(10)
            # Use predict() for output signature as predict_proba might not always be available/desired
            output_sample = pd.DataFrame(model.predict(input_sample), columns=['prediction'])
            signature = infer_signature(input_sample, output_sample)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_sample.to_dict(orient='split'), # Use dict format for input example
                pip_requirements="requirements.txt" # Link to project requirements
            )
            model_uri = f"runs:/{mlflow_run_id}/model"
            logger.info(f"Model logged. URI: {model_uri}")
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("training_status", "success")

            # Documentation for Error Handling Choice:
            # A check was added to ensure the dataframe size after filtering for relevant loan statuses
            # ('Fully Paid', 'Charged Off') is above the minimum threshold (min_records).
            # If the check fails, a ValueError is raised, causing this Prefect task to fail
            # and stopping the pipeline execution. This prevents attempting to train on
            # impractically small data. Failure details are logged to MLflow parameters.

        except Exception as e:
            logger.error(f"ERROR during training task: {e}", exc_info=True) # Log traceback
            mlflow.log_param("training_status", "failed_runtime_error")
            mlflow.log_param("error_message", str(e))
            raise e # Re-raise to ensure Prefect marks task as failed

    # Return the model URI, run_id, and feature lists needed by the robustness task
    return model_uri, mlflow_run_id, numeric_features, categorical_features

@task
def test_model_robustness_task(
    model_uri_to_test: str,
    data_path_for_test: str,
    mlflow_run_id_to_log_to: str,
    numeric_features: list, # Receive feature lists from training task
    categorical_features: list
    ):
    """
    Prefect task to load the trained model and perform a robustness check.
    Logs results back to the original MLflow run.
    Raises Exception on failure.
    """
    logger = get_run_logger()
    logger.info("--- Task: Model Robustness Test ---")

    if model_uri_to_test is None:
         logger.error("Cannot run robustness test: Invalid model URI provided.")
         raise ValueError("Invalid model URI received for robustness test.")

    logger.info(f"Loading model from URI: {model_uri_to_test}")
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri_to_test)
        logger.info("Model loaded successfully.")
    except Exception as e:
         logger.error(f"Failed to load model: {e}", exc_info=True)
         raise e # Fail the task

    try:
        # --- Robustness Test Definition (Justification) ---
        # Expectation: Model default predictions should show low sensitivity to minor
        # perturbations in key financial indicators (e.g., annual income, DTI, revol_util).
        # Justification: Tests stability against common real-world data noise. Large prediction
        # shifts for small input changes suggest overfitting or unreliability. Threshold (15%)
        # allows some sensitivity but flags excessive instability.

        logger.info("Preparing data for robustness test...")
        # Load data again for the test sample
        df_test_robust = pd.read_csv(data_path_for_test, low_memory=False)

        # --- Apply the *exact same* initial filtering and cleaning as in training ---
        valid_statuses = ['Fully Paid', 'Charged Off']
        df_test_robust = df_test_robust[df_test_robust['loan_status'].isin(valid_statuses)].copy()

        # Apply cleaning functions (ensure these match training!)
        if 'emp_length' in df_test_robust.columns:
            df_test_robust['emp_length_numeric'] = df_test_robust['emp_length'].apply(parse_emp_length)
        if 'int_rate' in df_test_robust.columns:
             df_test_robust['int_rate'] = df_test_robust['int_rate'].astype(str).str.rstrip('%').astype(float) / 100.0
        if 'revol_util' in df_test_robust.columns:
            df_test_robust['revol_util'] = df_test_robust['revol_util'].astype(str).str.rstrip('%')
            df_test_robust['revol_util'] = pd.to_numeric(df_test_robust['revol_util'], errors='coerce') / 100.0


        # Select the *same* features used for training
        features_to_use = numeric_features + categorical_features
        # Keep only necessary columns + target for potential context (optional)
        df_robust_sample_base = df_test_robust[features_to_use].copy()

        # Handle NaNs CONSISTENTLY with training (Drop on numeric, fill on categorical)
        # Note: SimpleImputer within the pipeline handles NaNs at predict time,
        # but we need non-NaN data *before* adding noise.
        # Drop rows where key numerics became NaN during loading/cleaning
        df_robust_sample_base.dropna(subset=numeric_features, inplace=True)
        for col in categorical_features:
            if df_robust_sample_base[col].isnull().any():
                df_robust_sample_base[col].fillna('Missing', inplace=True)

        # Take sample from the cleaned base data
        sample_size = min(1000, len(df_robust_sample_base))
        if sample_size == 0:
            logger.warning("No data available for robustness test after cleaning. Skipping.")
            # Log skip to MLflow?
            with mlflow.start_run(run_id=mlflow_run_id_to_log_to):
                mlflow.log_param("robustness_test_status", "skipped_no_data")
            return True # Or False? Decide if skipping is failure. Let's say True for now.

        X_sample = df_robust_sample_base.sample(n=sample_size, random_state=456)
        logger.info(f"Performing robustness check on {len(X_sample)} samples...")


        # --- Execute Robustness Test ---
        # 1. Predict on original sample
        # The loaded pyfunc model includes the preprocessing pipeline
        original_predictions = loaded_model.predict(X_sample)

        # 2. Create perturbed sample
        perturbed_sample = X_sample.copy()
        features_to_perturb = ['annual_inc', 'dti', 'revol_util'] # Key financial ratios/inputs
        noise_level_std_fraction = 0.05 # Noise std dev = 5% of feature's std dev

        for col in features_to_perturb:
            if col in perturbed_sample.columns and col in numeric_features: # Check it's a numeric feature we used
                std_dev = perturbed_sample[col].std()
                if pd.notna(std_dev) and std_dev > 0: # Ensure std_dev is valid
                     noise = np.random.normal(0, std_dev * noise_level_std_fraction, size=perturbed_sample.shape[0])
                     perturbed_sample[col] = perturbed_sample[col] + noise
                     # Clip values to logical bounds after adding noise
                     if col == 'dti': perturbed_sample[col] = perturbed_sample[col].clip(lower=0)
                     if col == 'annual_inc': perturbed_sample[col] = perturbed_sample[col].clip(lower=0)
                     if col == 'revol_util': perturbed_sample[col] = perturbed_sample[col].clip(lower=0, upper=max(1.5, perturbed_sample[col].max())) # Cap at 150% or observed max

        # 3. Predict on perturbed sample
        perturbed_predictions = loaded_model.predict(perturbed_sample)

        # 4. Compare predictions
        num_changed = np.sum(original_predictions != perturbed_predictions)
        change_percentage = (num_changed / len(X_sample)) * 100
        logger.info(f"Noise added to: {features_to_perturb}")
        logger.info(f"Number of predictions changed after adding noise: {num_changed} ({change_percentage:.2f}%)")


        # --- Evaluation of Robustness Test ---
        robustness_threshold = 15.0 # Allow up to 15% change
        passed_robustness = change_percentage < robustness_threshold
        logger.info(f"Robustness test passed (change < {robustness_threshold}%): {passed_robustness}")


        # --- Log results back to the *original* training run ---
        with mlflow.start_run(run_id=mlflow_run_id_to_log_to): # Re-open the original run
            mlflow.log_metric("robustness_change_percentage", change_percentage)
            mlflow.log_param("robustness_features_perturbed", ", ".join(features_to_perturb))
            mlflow.log_param("robustness_noise_level_std_fraction", noise_level_std_fraction)
            mlflow.log_param("robustness_threshold_percentage", robustness_threshold)
            mlflow.log_metric("robustness_test_passed", int(passed_robustness))
            mlflow.log_param("robustness_test_status", "completed") # Mark as completed

        if not passed_robustness:
            logger.warning("Model robustness test failed (change percentage exceeded threshold)!")
            # Decide if flow should fail: raise ValueError("Robustness test failed.")

        return passed_robustness # Return boolean status

    except Exception as e:
        logger.error(f"Error during robustness test execution: {e}", exc_info=True)
        # Log failure to MLflow run
        with mlflow.start_run(run_id=mlflow_run_id_to_log_to):
             mlflow.log_param("robustness_test_status", "failed_execution")
             mlflow.log_param("robustness_error", str(e))
        raise e # Fail the task


# --- Flow Definition ---

@flow(name="Lending Club Pipeline (Prefect)")
def lending_club_pipeline_prefect(
    data_path: str = "data/accepted_2007_to_2018Q4.csv", # Default path/name
    min_training_records: int = 5000,
    test_set_size: float = 0.2
):
    """
    Main Prefect flow orchestrating the Lending Club model pipeline:
    1. Data Quality Check
    2. Train & Version Model (predicting loan default)
    3. Model Robustness Test
    """
    logger = get_run_logger()
    logger.info("--- Starting Prefect Flow: Lending Club Pipeline ---")
    logger.info(f"Parameters: data_path='{data_path}', min_training_records={min_training_records}, test_set_size={test_set_size}")

    # Step 1: Data Quality Check
    # If this task fails (raises exception), the flow run will stop here.
    perform_data_quality_check_task(data_path=data_path)

    # Step 2: Train and Version Model
    # This task returns outputs needed by the next step. Prefect handles passing results.
    # If this task fails, the flow run will stop here.
    model_uri, train_run_id, numeric_features, categorical_features = train_and_version_model_task(
        data_path=data_path,
        min_records=min_training_records,
        test_set_prop=test_set_size
    )

    # Step 3: Model Robustness Test
    # This task depends on the outputs from the training task.
    # If this task fails, the flow run will stop here.
    test_model_robustness_task(
        model_uri_to_test=model_uri,
        data_path_for_test=data_path, # Use original data path for sampling test data
        mlflow_run_id_to_log_to=train_run_id,
        numeric_features=numeric_features, # Pass feature lists
        categorical_features=categorical_features
    )

    logger.info("--- Prefect Flow: Lending Club Pipeline Finished Successfully ---")


# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT: Set the correct path to your data file here ---
    actual_data_file = "data/accepted_2007_to_2018Q4.csv"

    # Scenario 1: Run with default settings
    print(f"Running pipeline with default settings (data: {actual_data_file})...")
    lending_club_pipeline_prefect(data_path=actual_data_file)

    # # Scenario 2: Test insufficient data error handling
    # print("Running pipeline to test insufficient data error...")
    # try:
    #     lending_club_pipeline_prefect(data_path=actual_data_file, min_training_records=50000000) # Set very high
    # except Exception as e:
    #     print(f"Pipeline failed as expected due to insufficient data: {e}")

    # # Scenario 3: Run with slightly different parameters
    # print("Running pipeline with modified test set size...")
    # lending_club_pipeline_prefect(data_path=actual_data_file, test_set_size=0.3)