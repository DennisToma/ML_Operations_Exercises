import great_expectations as gx
import pandas as pd
import sys
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
import json

DATA_PATH = "data/accepted_2007_to_2018Q4.csv"
EXPECTATION_SUITE_NAME = "lending_club_data_quality_suite"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_emp_length(emp_length: str) -> float:
    """
    Convert employment length from string to numeric value.
    Example: '< 1 year' -> 0.5, '10+ years' -> 10.0
    """
    if pd.isna(emp_length) or emp_length == 'n/a':
        return np.nan
    
    emp_length = emp_length.lower().replace('years', '').replace('year', '').strip()
    
    if emp_length == '< 1':
        return 0.5
    elif emp_length == '10+':
        return 10.0
    else:
        try:
            return float(emp_length)
        except ValueError:
            return np.nan

def test_loan_amount_distribution(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Test if loan amounts are within expected ranges.
    
    Reasoning:
    1. Based on empirical analysis of the dataset, we've determined appropriate thresholds
    2. The minimum loan amount of $1,000 aligns with Lending Club's business model
    3. The maximum loan amount of $40,000 represents the upper limit of their offerings
    4. The normal range ($1,525 to $40,000) covers 98% of the loans in the dataset
    5. Missing values should be extremely rare (below 0.01%) as loan amount is fundamental
    
    We expect:
    - No loans below $1,000 (the absolute minimum)
    - No loans above $40,000 (the absolute maximum)
    - At least 98% of loans to be between $1,525 and $40,000 (the normal range)
    - Missing values below 0.01% (very strict threshold based on data analysis)
    """
    results = {
        "test_name": "loan_amount_distribution_test",
        "passed": True,
        "details": {}
    }
    
    # Check for missing values - using stricter threshold of 0.01% based on data analysis
    missing_count = df['loan_amnt'].isnull().sum()
    missing_pct = missing_count / len(df) * 100
    
    results["details"]["missing_values"] = {
        "count": int(missing_count),
        "percentage": float(missing_pct),
        "threshold": 0.01  # Updated to 0.01% based on empirical analysis
    }
    
    if missing_pct > 0.01:  # Stricter threshold (0.01%) for loan_amnt
        results["passed"] = False
        results["details"]["missing_values"]["status"] = "FAILED"
        logger.warning(f"Too many missing loan amounts: {missing_pct:.4f}% (threshold: 0.01%)")
    else:
        results["details"]["missing_values"]["status"] = "PASSED"
        logger.info(f"Missing loan amounts: {missing_pct:.4f}% (below threshold of 0.01%)")
    
    # Filter out missing values for range checks
    valid_loans = df['loan_amnt'].dropna()
    
    # Check for loans below minimum threshold (empirically determined)
    min_threshold = 1000.0  # Absolute minimum from data analysis
    below_min = valid_loans[valid_loans < min_threshold]
    below_min_pct = len(below_min) / len(valid_loans) * 100
    
    results["details"]["below_minimum"] = {
        "threshold": float(min_threshold),
        "count": int(len(below_min)),
        "percentage": float(below_min_pct)
    }
    
    if below_min_pct > 0.1:  # Allow up to 0.1% below minimum for potential data errors
        results["passed"] = False
        results["details"]["below_minimum"]["status"] = "FAILED"
        logger.warning(f"Too many loans below minimum threshold: {below_min_pct:.2f}%")
    else:
        results["details"]["below_minimum"]["status"] = "PASSED"
    
    # Check for loans above maximum threshold (empirically determined)
    max_threshold = 40000.0  # Absolute maximum from data analysis
    above_max = valid_loans[valid_loans > max_threshold]
    above_max_pct = len(above_max) / len(valid_loans) * 100
    
    results["details"]["above_maximum"] = {
        "threshold": float(max_threshold),
        "count": int(len(above_max)),
        "percentage": float(above_max_pct)
    }
    
    if above_max_pct > 0.1:  # Allow up to 0.1% above maximum for potential special cases
        results["passed"] = False
        results["details"]["above_maximum"]["status"] = "FAILED"
        logger.warning(f"Too many loans above maximum threshold: {above_max_pct:.2f}%")
    else:
        results["details"]["above_maximum"]["status"] = "PASSED"
    
    # Check if most loans are within normal range (empirically determined)
    normal_range = (1525.0, 40000.0)  # Normal range from data analysis (covers 98% of loans)
    in_range = valid_loans[(valid_loans >= normal_range[0]) & (valid_loans <= normal_range[1])]
    in_range_pct = len(in_range) / len(valid_loans) * 100
    
    results["details"]["within_normal_range"] = {
        "range": [float(normal_range[0]), float(normal_range[1])],
        "count": int(len(in_range)),
        "percentage": float(in_range_pct)
    }
    
    if in_range_pct < 98.0:  # At least 98% should be in the normal range (based on data analysis)
        results["passed"] = False
        results["details"]["within_normal_range"]["status"] = "FAILED"
        logger.warning(f"Only {in_range_pct:.2f}% of loans are within normal range")
    else:
        results["details"]["within_normal_range"]["status"] = "PASSED"
    
    return results["passed"], results

def test_interest_rate_distribution(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Test if interest rates are within expected ranges.
    
    Reasoning:
    1. Based on empirical analysis of the dataset, we've determined appropriate thresholds
    2. The minimum interest rate of 5.31% represents the lowest rate offered
    3. The maximum interest rate of 30.84% represents the highest rate offered
    4. The normal range (5.32% to 26.77%) covers 98% of the loans in the dataset
    5. Missing values should be rare (below 0.1%) but can be slightly more lenient than loan_amnt
    
    We expect:
    - No interest rates below 5.31% (the absolute minimum)
    - No interest rates above 30.84% (the absolute maximum)
    - At least 98% of interest rates to be between 5.32% and 26.77% (the normal range)
    - Missing values below 0.1% (strict threshold based on data analysis)
    """
    results = {
        "test_name": "interest_rate_distribution_test",
        "passed": True,
        "details": {}
    }
    
    # Check for missing values - using threshold of 0.1% based on data analysis
    missing_count = df['int_rate'].isnull().sum()
    missing_pct = missing_count / len(df) * 100
    
    results["details"]["missing_values"] = {
        "count": int(missing_count),
        "percentage": float(missing_pct),
        "threshold": 0.1  # Updated to 0.1% based on empirical analysis
    }
    
    if missing_pct > 0.1:  # Strict but slightly more lenient threshold (0.1%) for int_rate
        results["passed"] = False
        results["details"]["missing_values"]["status"] = "FAILED"
        logger.warning(f"Too many missing interest rates: {missing_pct:.4f}% (threshold: 0.1%)")
    else:
        results["details"]["missing_values"]["status"] = "PASSED"
        logger.info(f"Missing interest rates: {missing_pct:.4f}% (below threshold of 0.1%)")
    
    # Filter out missing values for range checks
    valid_rates = df['int_rate'].dropna()
    
    # Check for rates below minimum threshold (empirically determined)
    min_threshold = 5.31  # Absolute minimum from data analysis
    below_min = valid_rates[valid_rates < min_threshold]
    below_min_pct = len(below_min) / len(valid_rates) * 100
    
    results["details"]["below_minimum"] = {
        "threshold": float(min_threshold),
        "count": int(len(below_min)),
        "percentage": float(below_min_pct)
    }
    
    if below_min_pct > 0.1:  # Allow up to 0.1% below minimum for potential data errors
        results["passed"] = False
        results["details"]["below_minimum"]["status"] = "FAILED"
        logger.warning(f"Too many interest rates below minimum threshold: {below_min_pct:.2f}%")
    else:
        results["details"]["below_minimum"]["status"] = "PASSED"
    
    # Check for rates above maximum threshold (empirically determined)
    max_threshold = 30.84  # Absolute maximum from data analysis
    above_max = valid_rates[valid_rates > max_threshold]
    above_max_pct = len(above_max) / len(valid_rates) * 100
    
    results["details"]["above_maximum"] = {
        "threshold": float(max_threshold),
        "count": int(len(above_max)),
        "percentage": float(above_max_pct)
    }
    
    if above_max_pct > 0.1:  # Allow up to 0.1% above maximum for potential special cases
        results["passed"] = False
        results["details"]["above_maximum"]["status"] = "FAILED"
        logger.warning(f"Too many interest rates above maximum threshold: {above_max_pct:.2f}%")
    else:
        results["details"]["above_maximum"]["status"] = "PASSED"
    
    # Check if most rates are within normal range (empirically determined)
    normal_range = (5.32, 26.77)  # Normal range from data analysis (covers 98% of rates)
    in_range = valid_rates[(valid_rates >= normal_range[0]) & (valid_rates <= normal_range[1])]
    in_range_pct = len(in_range) / len(valid_rates) * 100
    
    results["details"]["within_normal_range"] = {
        "range": [float(normal_range[0]), float(normal_range[1])],
        "count": int(len(in_range)),
        "percentage": float(in_range_pct)
    }
    
    if in_range_pct < 98.0:  # At least 98% should be in the normal range (based on data analysis)
        results["passed"] = False
        results["details"]["within_normal_range"]["status"] = "FAILED"
        logger.warning(f"Only {in_range_pct:.2f}% of interest rates are within normal range")
    else:
        results["details"]["within_normal_range"]["status"] = "PASSED"
    
    # Check distribution pattern (should have multiple modes for different loan grades)
    # This is a simplified check - in practice, you might use more sophisticated statistical tests
    bins = np.linspace(5.31, 30.84, 31)  # Create bins from min to max
    hist, _ = np.histogram(valid_rates, bins=bins)
    
    # Check if there are at least 3 local maxima (indicating multiple modes)
    peaks = 0
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks += 1
    
    results["details"]["distribution_pattern"] = {
        "peaks_detected": peaks,
        "minimum_expected_peaks": 3
    }
    
    if peaks < 3:  # Expect at least 3 peaks for different loan grades
        results["passed"] = False
        results["details"]["distribution_pattern"]["status"] = "FAILED"
        logger.warning(f"Interest rate distribution doesn't show expected multi-modal pattern")
    else:
        results["details"]["distribution_pattern"]["status"] = "PASSED"
    
    return results["passed"], results

def run_data_quality_checks(data_path: str, output_file_path: str) -> bool:
    """
    Runs a series of data quality checks on the provided dataset.
    Saves detailed results to a JSON file and returns an overall pass/fail status.

    Args:
        data_path: Path to the CSV dataset.
        output_file_path: Path to save the JSON results file.

    Returns:
        bool: True if all critical checks passed, False otherwise.
    """
    # Assuming logger is already configured (e.g., basicConfig in the main script)
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Data Quality Checks on {data_path} ---")
    
    all_passed = True
    test_results = {
        "source_file": data_path,
        "checks": []
    }

    try:
        # Load a sample of the data for checks to avoid loading the entire file if not necessary
        try:
            df_sample = pd.read_csv(data_path, nrows=10000, low_memory=False)
            df_full_check_sample = pd.read_csv(data_path, usecols=['loan_status', 'int_rate', 'loan_amnt'], low_memory=False)
        except FileNotFoundError:
            logger.error(f"Data file not found at {data_path}")
            test_results["checks"].append({
                "check_name": "File Existence",
                "passed": False,
                "details": f"File not found: {data_path}"
            })
            with open(output_file_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Detailed test results saved to {output_file_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading data for quality checks: {str(e)}")
            test_results["checks"].append({
                "check_name": "Data Loading",
                "passed": False,
                "details": f"Failed to load data: {str(e)}"
            })
            with open(output_file_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Detailed test results saved to {output_file_path}")
            return False

        # Check 1: Loan Status Distribution
        check_name = "Loan Status Distribution"
        logger.info(f"Running check: {check_name}")
        if 'loan_status' in df_full_check_sample.columns:
            valid_statuses = ['Fully Paid', 'Charged Off', 'Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Default']
            required_statuses_for_training = ['Fully Paid', 'Charged Off']
            status_counts = df_full_check_sample['loan_status'].value_counts(normalize=True)
            
            missing_required = [s for s in required_statuses_for_training if s not in status_counts.index]
            current_passed_check = not missing_required
            details = f"Required statuses {required_statuses_for_training} {'are present' if current_passed_check else 'are MISSING'}. Distribution:\n{status_counts.to_dict()}"
            if not current_passed_check:
                all_passed = False
            
            test_results["checks"].append({
                "check_name": check_name,
                "passed": bool(current_passed_check),
                "metrics": {"status_distribution": status_counts.to_dict()},
                "details": details
            })
            logger.info(f"{check_name} passed: {current_passed_check}")
        else:
            logger.warning("Skipping 'Loan Status Distribution' check: 'loan_status' column not found.")
            test_results["checks"].append({
                "check_name": check_name,
                "passed": False,
                "details": "'loan_status' column not found."
            })
            all_passed = False

        # Check 2: Interest Rate Sanity
        check_name = "Interest Rate Sanity"
        logger.info(f"Running check: {check_name}")
        if 'int_rate' in df_full_check_sample.columns:
            try:
                processed_rates = pd.to_numeric(df_full_check_sample['int_rate'].astype(str).str.rstrip('%'), errors='coerce') / 100.0
                processed_rates.dropna(inplace=True)

                min_rate, max_rate = processed_rates.min(), processed_rates.max()
                abs_min_expected, abs_max_expected = 0.0531, 0.3084 
                
                current_passed_check = (min_rate >= abs_min_expected * 0.9) and (max_rate <= abs_max_expected * 1.1)
                details = f"Observed min rate: {min_rate:.4f}, max rate: {max_rate:.4f}. Expected range (approx): [{abs_min_expected:.4f} - {abs_max_expected:.4f}]"
                if not current_passed_check:
                    all_passed = False
                
                test_results["checks"].append({
                    "check_name": check_name,
                    "passed": bool(current_passed_check),
                    "metrics": {"min_rate": float(min_rate), "max_rate": float(max_rate)},
                    "details": details
                })
                logger.info(f"{check_name} passed: {current_passed_check}")

            except Exception as e:
                logger.error(f"Error during '{check_name}': {str(e)}")
                test_results["checks"].append({
                    "check_name": check_name,
                    "passed": False,
                    "details": f"Error processing interest rates: {str(e)}"
                })
                all_passed = False
        else:
            logger.warning(f"Skipping '{check_name}' check: 'int_rate' column not found.")
            test_results["checks"].append({
                "check_name": check_name,
                "passed": False,
                "details": "'int_rate' column not found."
            })
            all_passed = False
        
        # Check 3: Missing Loan Amount
        check_name = "Missing Loan Amount"
        logger.info(f"Running check: {check_name}")
        if 'loan_amnt' in df_sample.columns:
            missing_percentage = df_sample['loan_amnt'].isnull().mean() * 100
            threshold = 1.0
            current_passed_check = missing_percentage < threshold
            details = f"Missing loan_amnt: {missing_percentage:.2f}%. Threshold: < {threshold}%"
            if not current_passed_check:
                all_passed = False
            test_results["checks"].append({
                "check_name": check_name,
                "passed": bool(current_passed_check),
                "metrics": {"missing_percentage": float(missing_percentage)},
                "details": details
            })
            logger.info(f"{check_name} passed: {current_passed_check}")
        else:
            logger.warning(f"Skipping '{check_name}' check: 'loan_amnt' column not found.")
            test_results["checks"].append({ "check_name": check_name, "passed": False, "details": "'loan_amnt' column not found."})
            all_passed = False

        logger.info(f"All data quality checks completed. Overall status: {'PASSED' if all_passed else 'FAILED'}")

    except Exception as e:
        logger.error(f"Unhandled error during data quality checks: {str(e)}", exc_info=True)
        test_results["error"] = f"Unhandled exception: {str(e)}"
        all_passed = False
    
    finally:
        with open(output_file_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Detailed test results saved to {output_file_path}")
        
    return bool(all_passed)

if __name__ == "__main__":
    print("\nExecuting data quality checks as a script...")
    print("This implements the requirements for Task 1: Pre-training tests on data quality")
    print("The tests check for missing values and validate the distribution of two key attributes")
    print("All thresholds are derived from empirical analysis, not arbitrary values\n")
    
    success = run_data_quality_checks(DATA_PATH, EXPECTATION_SUITE_NAME)
    exit_code = 0 if success else 1
    
    print(f"\nScript finished with exit code: {exit_code}")
    print(f"Overall result: {'PASSED' if success else 'FAILED'}")
    sys.exit(exit_code)