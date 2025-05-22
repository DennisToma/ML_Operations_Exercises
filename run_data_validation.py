# run_data_validation.py
import os
import sys
import json
import logging
import pathlib

# Add current directory to sys.path to allow importing data_quality_tests
# This is important if data_quality_tests.py is in the same directory
# and the script is run as /app/run_data_validation.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_quality_tests import run_data_quality_checks, parse_emp_length # parse_emp_length might not be needed here but good to check import
except ImportError as e:
    logging.error(f"Failed to import from data_quality_tests: {e}")
    logging.error("Ensure data_quality_tests.py is in the same directory as run_data_validation.py or in PYTHONPATH.")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Data Validation Script ---")

    data_path = os.getenv("DATA_PATH")
    output_dir = os.getenv("OUTPUT_DIR", "/app/outputs") # Default if not set

    if not data_path:
        logger.error("DATA_PATH environment variable not set.")
        sys.exit(1)

    # Ensure output directory exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_file_path = os.path.join(output_dir, "data_quality_report.json")

    logger.info(f"Data path: {data_path}")
    logger.info(f"Output report will be saved to: {report_file_path}")

    try:
        checks_passed = run_data_quality_checks(data_path=data_path, output_file_path=report_file_path)

        if checks_passed:
            logger.info("Data quality checks passed successfully.")
            sys.exit(0) # Success
        else:
            logger.error("Data quality checks failed. See report for details.")
            sys.exit(1) # Failure
            
    except Exception as e:
        logger.error(f"An unexpected error occurred during data validation: {e}", exc_info=True)
        # Save a minimal error report if possible
        error_report = {
            "status": "SCRIPT_ERROR",
            "error_message": str(e),
            "data_path": data_path
        }
        with open(report_file_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        logger.info(f"Error report saved to {report_file_path}")
        sys.exit(1) # Failure

if __name__ == "__main__":
    main()