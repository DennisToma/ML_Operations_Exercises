import os
import sys
import json
import logging
import pathlib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import collections

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_drift_checks(data_path: str, output_file_path: str, test_fraction: float = 0.1):
    """
    Performs covariate drift detection on a dataset by comparing a reference
    segment (the first N%) with a current segment (the last N%).

    This simulates monitoring for drift in new, incoming data that was not
    used for training. The choice of using the last 10% of the data as the
    "current" set is a proxy for observing production data over time. The first
    10% serves as a stable reference baseline, representing the data distribution
    the model was originally trained on.

    The test uses the two-sample Kolmogorov-Smirnov (K-S) test, which is a
    non-parametric test that compares the cumulative distributions of two data
    samples. It is sensitive to differences in both location and shape of the
    empirical distribution functions of the two samples.

    The "expected" behavior is that the distributions are similar, resulting in a
    p-value > 0.05. A p-value below this threshold suggests a statistically
    significant drift in the feature's distribution.

    Args:
        data_path (str): Path to the full dataset CSV.
        output_file_path (str): Path to save the JSON drift report.
        test_fraction (float): The fraction of data to use for reference and current sets.
    """
    logger.info("--- Starting Covariate Drift Monitoring ---")
    report = {
        "data_path": data_path,
        "test_fraction": test_fraction,
        "features": {}
    }
    overall_drift_detected = False

    try:
        # Memory-efficient way to compare early vs. late data using chunking,
        # which avoids loading the entire large file into memory.
        logger.info(f"Starting memory-efficient data loading from {data_path}...")
        
        chunk_size = 50000
        # Use a fixed number of chunks for the reference and current sets.
        # This provides a substantial sample size (e.g., 100k rows) for comparison
        # without overwhelming memory.
        num_chunks_for_set = 2 

        reference_chunks = []
        # A deque with a maxlen is a highly efficient way to keep track of the last N items.
        current_chunks_deque = collections.deque(maxlen=num_chunks_for_set)
        
        iterator = pd.read_csv(data_path, chunksize=chunk_size, low_memory=False)
        
        chunk_count = 0
        for chunk in iterator:
            if chunk_count < num_chunks_for_set:
                reference_chunks.append(chunk)
            current_chunks_deque.append(chunk)
            chunk_count += 1
        
        if chunk_count < num_chunks_for_set * 2: # Need at least enough chunks for two distinct sets
            logger.warning("Dataset is too small for a meaningful drift check with current chunk settings. Skipping.")
            report['status'] = "SKIPPED_TOO_SMALL"
            return
            
        reference_df = pd.concat(reference_chunks, ignore_index=True)
        current_df = pd.concat(list(current_chunks_deque), ignore_index=True)

        logger.info(f"Reference data size: {len(reference_df)} rows")
        logger.info(f"Current data size: {len(current_df)} rows")
        # Define reference and current datasets
        
        # Features to check for drift
        features_to_check = [
            'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
            'revol_bal', 'revol_util', 'total_acc'
        ]

        for feature in features_to_check:
            logger.info(f"Checking drift for feature: {feature}")
            
            # Clean and prepare data for the test
            ref_data = reference_df[feature].dropna()
            cur_data = current_df[feature].dropna()

            if ref_data.empty or cur_data.empty:
                logger.warning(f"Skipping {feature} due to no valid data in one of the sets.")
                continue
            
            # For percentage string columns
            if ref_data.dtype == 'object':
                ref_data = pd.to_numeric(ref_data.astype(str).str.rstrip('%'), errors='coerce').dropna()
            if cur_data.dtype == 'object':
                cur_data = pd.to_numeric(cur_data.astype(str).str.rstrip('%'), errors='coerce').dropna()

            # Perform the K-S test
            ks_statistic, p_value = ks_2samp(ref_data, cur_data)
            
            drift_detected = p_value < 0.05
            if drift_detected:
                overall_drift_detected = True

            report['features'][feature] = {
                "ks_statistic": ks_statistic,
                "p_value": p_value,
                "drift_detected": bool(drift_detected)
            }
            logger.info(f"  -> K-S Statistic: {ks_statistic:.4f}, P-Value: {p_value:.4f}, Drift: {drift_detected}")

        report['overall_drift_status'] = "DRIFT_DETECTED" if overall_drift_detected else "NO_DRIFT"
        logger.info(f"Overall Drift Status: {report['overall_drift_status']}")

    except Exception as e:
        logger.error(f"An error occurred during drift detection: {e}", exc_info=True)
        report['error'] = str(e)
    
    finally:
        logger.info(f"Saving drift report to {output_file_path}")
        pathlib.Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    data_path = os.getenv("DATA_PATH")
    output_dir = os.getenv("OUTPUT_DIR", "/app/outputs")
    
    if not data_path:
        logger.error("DATA_PATH environment variable not set.")
        sys.exit(1)
        
    report_file_path = os.path.join(output_dir, "drift_report.json")
    
    run_drift_checks(data_path, report_file_path)
    logger.info("Drift monitoring script finished.")

if __name__ == "__main__":
    main() 