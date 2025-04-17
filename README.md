# Lending Club Default Prediction Pipeline

This project implements a machine learning pipeline for predicting loan defaults using the Lending Club dataset. It includes data quality checks, preprocessing, model training, and evaluation.

Repository: https://github.com/DennisToma/Ex1

## Project Structure

- `pipeline.py`: Main ML pipeline script with MLflow and Prefect integration
- `data_quality_tests.py`: Pre-training data quality tests
- `Dockerfile`: Container definition for reproducible execution
- `requirements.txt`: Python dependencies

## Data Quality Tests

The `data_quality_tests.py` script implements pre-training data quality checks to ensure the dataset meets expected standards before model training. These tests were designed based on empirical analysis of the Lending Club dataset.

### Tests Implemented

1. **Loan Amount Distribution Test**
   - Checks if loan amounts fall within expected ranges
   - Minimum threshold: $1,000 (absolute minimum)
   - Maximum threshold: $40,000 (absolute maximum)
   - Normal range: $1,525 to $40,000 (should contain 98% of loans)
   - Validates that missing values are below 0.01% (very strict threshold)

2. **Interest Rate Distribution Test**
   - Checks if interest rates fall within expected ranges
   - Minimum threshold: 5.31% (absolute minimum)
   - Maximum threshold: 30.84% (absolute maximum)
   - Normal range: 5.32% to 26.77% (should contain 98% of rates)
   - Validates that the distribution shows multiple modes (at least 3 peaks)
   - Validates that missing values are below 0.1% (strict threshold)

### Threshold Determination

The thresholds used in these tests were derived from empirical analysis of the dataset:

1. I analyzed the distribution of loan amounts and interest rates
2. I calculated percentiles to understand the natural boundaries of the data
3. I performed a train-test split to validate that the distributions are consistent
4. I combined this empirical analysis with business knowledge (like Lending Club's loan limits)

#### Missing Value Thresholds

The missing value thresholds were specifically determined through detailed analysis:

| Attribute | Current Missing (%) | Recommended Threshold (%) | Strictness Level |
|-----------|---------------------|---------------------------|------------------|
| loan_amnt | 0.00146             | 0.01                      | very strict      |
| int_rate  | 0.00146             | 0.10                      | strict           |

**Reasoning for threshold selection:**

1. **Loan Amount:**
   - Fundamental attribute of any loan record
   - Critical for risk assessment and portfolio analysis
   - Should be available for virtually all records
   - Stricter threshold applied due to business importance

2. **Interest Rate:**
   - Important for understanding loan pricing and risk
   - May legitimately be missing for certain loan types or statuses
   - Slightly more lenient threshold compared to loan amount
   - Still requires high data quality for accurate analysis

These thresholds are based on:
- Empirical analysis of the current dataset
- Business importance of each attribute
- Industry standards for data quality
- Practical considerations for data processing

## Running the Project

### Using Docker (Recommended)

1. **Build the Docker image**:
   ```bash
   docker build -t lending-club-pipeline .
   ```

2. **Run the container**:
   
   On Linux/macOS:
   ```bash
   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/mlruns:/app/mlruns" lending-club-pipeline
   ```
   
   On Windows PowerShell:
   ```powershell
   docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/mlruns:/app/mlruns" lending-club-pipeline
   ```

3. **Run only the data quality tests**:
   ```bash
   docker run --rm -v "$(pwd)/data:/app/data" lending-club-pipeline python data_quality_tests.py
   ```

## Data Requirements

The pipeline expects the Lending Club dataset in CSV format:
- Main file: `data/accepted_2007_to_2018Q4.csv`

## Output

- Data quality test results are logged to the console
- Detailed test results are saved to `data_quality_test_results.json`
- MLflow tracking information is stored in the `mlruns` directory
- Model artifacts are saved in the MLflow registry

## Dependencies

Key dependencies include:
- pandas, numpy, scikit-learn: Data processing and modeling
- mlflow: Model tracking and registry
- prefect: Workflow orchestration
- great-expectations: Data quality testing

See `requirements.txt` for the complete list of dependencies.