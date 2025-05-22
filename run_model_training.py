# run_model_training.py
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
from sklearn.pipeline import Pipeline as SklearnPipeline # Renamed to avoid conflict
from sklearn.impute import SimpleImputer
import joblib
import mlflow
from mlflow.models import infer_signature
import psutil
import gc
import json
import logging
import sklearn
# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_quality_tests import parse_emp_length # Only need parse_emp_length
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

def train_model(
    data_path: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    output_dir: str,
    min_records: int,
    test_set_prop: float,
    sample_size: int = None
):
    logger.info("--- Starting Model Training Script ---")
    logger.info(f"Available memory before starting: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    logger.info(f"MLflow Experiment Name: {mlflow_experiment_name}")

    model_uri_output = None
    mlflow_run_id_output = None
    
    # Ensure output directory exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    meta_file_path = os.path.join(output_dir, "model_meta.json")

    active_run = None
    try:
        with mlflow.start_run(run_name="dockerized_training_run") as active_run:
            mlflow_run_id_output = active_run.info.run_id
            logger.info(f"MLflow Run ID: {mlflow_run_id_output}")
            mlflow.log_param("script_name", "run_model_training.py")
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("min_training_records_threshold", min_records)
            mlflow.log_param("test_set_prop", test_set_prop)
            if sample_size:
                mlflow.log_param("sample_size", sample_size)

            # --- Load Data ---
            logger.info(f"Loading data from {data_path}")
            chunk_size = 100000
            chunks = []
            valid_statuses = ['Fully Paid', 'Charged Off'] # Focus on binary classification for this model
            
            for i, chunk_df in enumerate(pd.read_csv(data_path, chunksize=chunk_size, low_memory=True)):
                logger.info(f"Processing chunk {i+1}")
                filtered_chunk = chunk_df[chunk_df['loan_status'].isin(valid_statuses)].copy()
                if not filtered_chunk.empty:
                    chunks.append(filtered_chunk)
                del chunk_df, filtered_chunk
                gc.collect()
                if sample_size and sum(len(c) for c in chunks) >= sample_size:
                    logger.info(f"Reached target sample size of {sample_size}, stopping chunk loading.")
                    break
            
            if not chunks:
                error_message = "No data available after filtering for valid loan statuses."
                logger.error(error_message)
                mlflow.log_param("training_status", "failed_no_data_after_filter")
                mlflow.log_param("error_message", error_message)
                raise ValueError(error_message)

            df = pd.concat(chunks, ignore_index=True)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Memory after loading: {psutil.virtual_memory().available / (1024**3):.2f} GB")

            # --- Preprocessing, Target Definition ---
            df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
            if 'emp_length' in df.columns:
                df['emp_length_numeric'] = df['emp_length'].apply(parse_emp_length)
            if 'int_rate' in df.columns:
                df['int_rate'] = df['int_rate'].astype(str).str.rstrip('%').astype(float) / 100.0
            if 'revol_util' in df.columns:
                df['revol_util'] = df['revol_util'].astype(str).str.rstrip('%')
                df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce') / 100.0

            if len(df) < min_records:
                error_message = f"Data size ({len(df)}) after initial processing is less than minimum ({min_records})."
                logger.error(error_message)
                mlflow.log_param("training_status", "failed_insufficient_data")
                mlflow.log_param("error_message", error_message)
                raise ValueError(error_message)

            # --- Feature Selection ---
            numeric_features = [
                'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
                'revol_bal', 'revol_util', 'total_acc'
            ]
            if 'emp_length_numeric' in df.columns:
                numeric_features.append('emp_length_numeric')
            categorical_features = [
                'term', 'grade', 'sub_grade', 'home_ownership', 
                'verification_status', 'purpose', 'addr_state'
            ]
            features_to_use = numeric_features + categorical_features
            mlflow.log_param("numeric_features_used", ", ".join(numeric_features))
            mlflow.log_param("categorical_features_used", ", ".join(categorical_features))

            df.dropna(subset=numeric_features + ['target'], inplace=True) # Drop NaNs in key numerics and target
            for col in categorical_features: # Impute categoricals
                if df[col].isnull().any(): # Check if imputation is needed
                    df[col].fillna('Missing', inplace=True)
            
            logger.info(f"Data shape after cleaning/NaN handling: {df.shape}")
            if len(df) < min_records: # Check again after NaN handling
                error_message = f"Data size ({len(df)}) after NaN handling is less than minimum ({min_records})."
                logger.error(error_message)
                mlflow.log_param("training_status", "failed_insufficient_data_post_nan")
                mlflow.log_param("error_message", error_message)
                raise ValueError(error_message)

            X = df[features_to_use]
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_set_prop, random_state=42, stratify=y
            )
            mlflow.log_param("train_set_size", len(X_train))
            mlflow.log_param("test_set_actual_size", len(X_test))
            del df; gc.collect()

            # --- Preprocessing & Model Pipeline ---
            numeric_transformer = SklearnPipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            categorical_transformer = SklearnPipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)],
                remainder='drop')
            
            model_pipeline = SklearnPipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', solver='liblinear'))
            ])

            logger.info("Training the model...")
            model_pipeline.fit(X_train, y_train)
            logger.info("Model training complete.")

            # --- Evaluation ---
            y_pred_test = model_pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            mlflow.log_metric("test_accuracy", test_accuracy)
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            try:
                y_proba_test = model_pipeline.predict_proba(X_test)[:, 1]
                test_roc_auc = roc_auc_score(y_test, y_proba_test)
                mlflow.log_metric("test_roc_auc", test_roc_auc)
                logger.info(f"Test ROC AUC: {test_roc_auc:.4f}")
            except AttributeError:
                logger.warning("Model does not support predict_proba, skipping ROC AUC.")
            
            report_str = classification_report(y_test, y_pred_test, target_names=['Fully Paid', 'Charged Off'], zero_division=0)
            logger.info(f"\nClassification Report:\n{report_str}")
            mlflow.log_text(report_str, "classification_report.txt")

            # --- Serialize & Version Model ---
            logger.info("Serializing and registering model with MLflow...")
            signature = None
            if not X_test.empty:
                 signature = infer_signature(X_test.iloc[:100], y_pred_test[:100] if len(y_pred_test) >= 100 else y_pred_test)

            # Simplified requirements for Docker context
            pip_requirements = [
                f"scikit-learn=={sklearn.__version__}", # Corrected line
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
                f"mlflow=={mlflow.__version__}"
            ]
            
            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_test.iloc[:5] if not X_test.empty else None,
                registered_model_name="LendingClubDefaultClassifier_Docker", # Potentially different name for dockerized versions
                pip_requirements=pip_requirements
            )
            model_uri_output = f"runs:/{mlflow_run_id_output}/model"
            logger.info(f"Model registered. URI: {model_uri_output}")
            mlflow.log_param("training_status", "success")
            
            # Save metadata for the next step
            model_meta = {
                "model_uri": model_uri_output,
                "mlflow_run_id": mlflow_run_id_output,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features
            }
            with open(meta_file_path, 'w') as f:
                json.dump(model_meta, f, indent=2)
            logger.info(f"Model metadata saved to {meta_file_path}")

            return True # Success

    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        if mlflow.active_run():
            mlflow.log_param("training_status", "failed_exception")
            mlflow.log_param("error_message", str(e))
        # Save partial metadata if possible, indicating failure
        error_meta = { "status": "TRAINING_FAILED", "error": str(e) }
        with open(meta_file_path, 'w') as f:
            json.dump(error_meta, f, indent=2)
        return False # Failure
    finally:
        if active_run: # Should be handled by 'with' statement, but as a safeguard
            mlflow.end_run()

def main():
    data_path = os.getenv("DATA_PATH")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "LendingClubDockerized")
    output_dir = os.getenv("OUTPUT_DIR", "/app/outputs")
    
    min_records = int(os.getenv("MIN_TRAINING_RECORDS", "5000"))
    test_set_prop = float(os.getenv("TEST_SET_SIZE", "0.2"))
    use_sample_str = os.getenv("USE_SAMPLE", "False").lower()
    use_sample = use_sample_str == 'true'
    sample_size = int(os.getenv("SAMPLE_SIZE", "100000")) if use_sample else None

    if not data_path:
        logger.error("DATA_PATH environment variable not set.")
        sys.exit(1)

    success = train_model(
        data_path, mlflow_tracking_uri, mlflow_experiment_name, output_dir,
        min_records, test_set_prop, sample_size
    )

    if success:
        logger.info("Model training script finished successfully.")
        sys.exit(0)
    else:
        logger.error("Model training script failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()