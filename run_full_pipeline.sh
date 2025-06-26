#!/bin/bash
# This script orchestrates the entire MLOps pipeline, from data validation
# and model training to A/B testing, all in one command.

# Exit immediately if a command exits with a non-zero status. This ensures
# that the script will stop if any of the docker-compose stages fail.
set -e

# A helper function to run a Docker Compose stage
run_compose_stage() {
    # Takes the compose file path (optional) and stage name as arguments
    local compose_file="$1"
    local stage_name="$2"
    
    echo ""
    echo "--- Starting Stage: $stage_name ---"
    
    # Base command
    local command="docker-compose"
    
    # Add the file option if a specific file is provided
    if [ -n "$compose_file" ]; then
        command+=" -f $compose_file"
    fi
    
    # Add flags to build images and stop all containers when the target service exits
    command+=" up --build --abort-on-container-exit"
    
    echo "Executing command: $command"
    
    # Execute the command. The `eval` is safe here as we are constructing the command
    # from controlled, known-safe inputs within this script.
    eval "$command"
    
    echo "--- Stage '$stage_name' completed successfully. ---"
}

# --- STAGE 1: Train Model V1 and Run Drift Monitoring ---
run_compose_stage "docker-compose.yml" "V1 Model Training & Drift Monitoring"

# --- STAGE 2: Train Model V2 ---
run_compose_stage "docker-compose-v2.yml" "V2 Model Training"

# --- STAGE 3: Extract IDs and Run A/B Test ---
echo ""
echo "--- Preparing for Stage: A/B Test ---"

# Check if jq (a command-line JSON processor) is installed, as it's needed to parse the metadata files.
if ! command -v jq &> /dev/null
then
    echo "[ERROR] 'jq' is not installed, but it's required to parse metadata files."
    echo "Please install it to continue."
    echo "  - On Debian/Ubuntu: sudo apt-get install jq"
    echo "  - On macOS (with Homebrew): brew install jq"
    exit 1
fi

# Define paths to the metadata files
V1_META_PATH="outputs/model_meta_Model_V1_Default.json"
V2_META_PATH="outputs/model_meta_Model_V2_Low_Regularization.json"

# Check that the files exist before trying to parse them
if [ ! -f "$V1_META_PATH" ]; then
    echo "Error: Could not find V1 metadata file at '$V1_META_PATH'. The V1 training stage may have failed. Aborting."
    exit 1
fi
if [ ! -f "$V2_META_PATH" ]; then
    echo "Error: Could not find V2 metadata file at '$V2_META_PATH'. The V2 training stage may have failed. Aborting."
    exit 1
fi

# Use jq to extract the 'mlflow_run_id' from each JSON file and export it as an environment variable.
# Docker Compose automatically picks up environment variables from the shell it's run in.
export MODEL_V1_RUN_ID=$(jq -r '.mlflow_run_id' "$V1_META_PATH")
export MODEL_V2_RUN_ID=$(jq -r '.mlflow_run_id' "$V2_META_PATH")

# Validate that the IDs were extracted correctly
if [ -z "$MODEL_V1_RUN_ID" ] || [ "$MODEL_V1_RUN_ID" == "null" ]; then
    echo "Error: Failed to extract mlflow_run_id from '$V1_META_PATH'. Aborting."
    exit 1
fi
if [ -z "$MODEL_V2_RUN_ID" ] || [ "$MODEL_V2_RUN_ID" == "null" ]; then
    echo "Error: Failed to extract mlflow_run_id from '$V2_META_PATH'. Aborting."
    exit 1
fi

echo "Successfully extracted MLflow Run IDs:"
echo "  - Model V1 Run ID: $MODEL_V1_RUN_ID"
echo "  - Model V2 Run ID: $MODEL_V2_RUN_ID"

# Run the final A/B test stage
run_compose_stage "docker-compose-ab-test.yml" "A/B Model Comparison"

echo ""
echo "==============================================="
echo "--- Full Pipeline Completed Successfully! ---"
echo "===============================================" 