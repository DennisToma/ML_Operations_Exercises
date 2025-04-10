import great_expectations as gx
import pandas as pd
import sys
import numpy as np

DATA_PATH = "data/accepted_2007_to_2018Q4.csv"
EXPECTATION_SUITE_NAME = "retail_sales_suite_v1"

def parse_emp_length(length):
    """Helper to convert emp_length string to numeric."""
    if pd.isna(length):
        return np.nan # Keep NaN as NaN
    elif '10+' in length:
        return 10
    elif '< 1' in length:
        return 0
    else:
        try:
            # Extract digits
            return int(''.join(filter(str.isdigit, length)))
        except ValueError:
            return np.nan # Handle unexpected formats

def run_data_quality_checks(data_path=DATA_PATH):
    """
    Loads Lending Club data, defines expectations, runs validation, returns status.
    Returns:
        bool: True if validation passes, False otherwise.
    """
    print(f"--- Running Data Quality Checks on {data_path} ---")
    try:
        df = pd.read_csv(data_path, low_memory=False)

        # --- Basic Cleaning specific to Lending Club ---
        # Convert percentage strings to numeric
        if 'int_rate' in df.columns:
             df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').astype(float) / 100.0
        if 'revol_util' in df.columns:
             df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '').astype(float) / 100.0

        # Parse employment length
        if 'emp_length' in df.columns:
            df['emp_length_numeric'] = df['emp_length'].apply(parse_emp_length)

        print("Performed basic cleaning (rate/util conversion, emp_length parsing).")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return False
    except Exception as e:
        print(f"Error loading or cleaning data: {e}")
        return False

    context = gx.get_context(project_root_dir=None)
    datasource = context.sources.add_pandas("my_pandas_datasource")
    # Use the cleaned dataframe
    data_asset = datasource.add_dataframe_asset("lending_club_data", dataframe=df)
    suite = context.add_or_update_expectation_suite(expectation_suite_name=EXPECTATION_SUITE_NAME)

    # --- Add Expectations with Justifications ---

    # Test 1: Missing/Null Values in a critical feature
    # Expectation: Annual Income ('annual_inc') should have very few missing values (< 5%).
    # Justification: Annual income is fundamental for assessing a borrower's repayment ability (debt-to-income ratio).
    # A high percentage of missing values (>5%) would significantly impair the dataset's usability for credit risk
    # modeling and potentially indicate data collection issues. The 5% threshold allows for some minor gaps but flags systemic problems.
    suite.add_expectation(
    gx.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null", # <-- CORRECTED TYPE
        kwargs={
            "column": "annual_inc",
            "mostly": 0.95  # <-- This now means "at least 95% should NOT be null"
        },
        meta={ "notes": { "format": "markdown", "content": """**Expectation:** At least 95% of 'annual_inc' values must not be null (equivalent to < 5% nulls). **Justification:** Crucial for credit risk assessment (e.g., DTI calculation). Significant null rate (>5%) hinders modeling and suggests data quality issues.""" } }
    )
    )
    print("  Added: Expectation for non-null proportion in 'annual_inc'.")

    # Test 2: Attribute Distributions

    # Attribute 1: Loan Grade (Categorical)
    # Expectation: The 'grade' column must only contain the official Lending Club grades (A-G).
    # Justification: Loan grades are discrete risk categories assigned by Lending Club. Values outside this known set ('A', 'B', 'C', 'D', 'E', 'F', 'G') represent data corruption or errors, invalidating analyses based on grade. This set is derived from Lending Club's standard practice and dataset documentation.
    valid_grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    suite.add_expectation(
         gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "grade", "value_set": valid_grades},
            meta={ "notes": { "format": "markdown", "content": """**Expectation:** 'grade' must be in {'A', 'B', 'C', 'D', 'E', 'F', 'G'}. **Justification:** Represents defined risk categories by LC. Values outside this set indicate data errors/corruption, invalidating grade-based analysis.""" } }
        )
    )
    print("  Added: Expectation for value set in 'grade'.")


    # Attribute 2: Interest Rate (Numeric)
    # Expectation: Values in the 'int_rate' column (after conversion to decimal) should be between 0 and 0.5 (0% to 50%).
    # Justification: Interest rates must be non-negative. While rates vary, rates above 50% are highly improbable for this type of consumer lending and likely indicate data entry errors or outliers needing investigation. The 0% lower bound is theoretically possible (promotions), while 50% provides a reasonable upper sanity check based on typical market conditions for such loans. This test helps catch gross errors that could skew analysis.
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "int_rate",
                "min_value": 0.0,   # 0%
                "max_value": 0.5,   # 50% (Adjust if domain knowledge suggests different extreme)
                "mostly": 1.0       # Require ALL values within range after cleaning
            },
             meta={ "notes": { "format": "markdown", "content": """**Expectation:** 'int_rate' (decimal) between 0.0 and 0.5 (0%-50%). **Justification:** Rates must be non-negative. Rates >50% are highly unlikely for LC loans and suggest errors. Provides a sanity check against gross outliers skewing analysis.""" } }
        )
    )
    print("  Added: Expectation for value range in 'int_rate'.")

    # --- Save & Run Validation ---
    context.add_or_update_expectation_suite(expectation_suite=suite)
    checkpoint = context.add_or_update_checkpoint(
        name="lending_club_dq_checkpoint",
        validations=[{"batch_request": data_asset.build_batch_request(), "expectation_suite_name": EXPECTATION_SUITE_NAME}]
    )
    checkpoint_result = checkpoint.run()

    print(f"--- Data Quality Check Complete. Success: {checkpoint_result.success} ---")
    if not checkpoint_result.success:
         print("Validation Failed. Check details if needed.") # More detailed logging can be added
    return checkpoint_result.success

if __name__ == "__main__":
    print("Executing data quality checks as a script...")
    success = run_data_quality_checks()
    exit_code = 0 if success else 1
    print(f"Script finished with exit code: {exit_code}")
    sys.exit(exit_code)