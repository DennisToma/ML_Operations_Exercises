# run_full_pipeline.ps1
# This script orchestrates the entire MLOps pipeline, from data validation
# and model training to A/B testing, all in one command.

# Stop on any error
$ErrorActionPreference = "Stop"

# A helper function to run a Docker Compose stage and check its exit code
function Run-Compose-Stage {
    param(
        [string]$ComposeFile,
        [string]$StageName
    )
    # The --abort-on-container-exit flag ensures that all containers are stopped
    # as soon as any container exits, making the 'up' command terminate after the
    # final service in the dependency chain finishes.
    Write-Host "`n--- Starting Stage: $StageName ---"
    
    $command = "docker-compose"
    if ($ComposeFile) {
        $command += " -f $ComposeFile"
    }
    $command += " up --build --abort-on-container-exit"
    
    Write-Host "Executing command: $command"
    
    # Execute the command
    Invoke-Expression $command
    
    # Check the exit code of the last command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Stage '$StageName' failed with exit code $LASTEXITCODE. Aborting pipeline."
        # The script will stop here due to $ErrorActionPreference = "Stop"
        exit 1
    } else {
        Write-Host "--- Stage '$StageName' completed successfully. ---"
    }
}

# --- STAGE 1: Train Model V1 and Run Drift Monitoring ---
# This uses the main docker-compose.yml file. It will build and run all services,
# respecting the `depends_on` order. It will produce the V1 model and its metadata.
Run-Compose-Stage -ComposeFile "docker-compose.yml" -StageName "V1 Model Training & Drift Monitoring"

# --- STAGE 2: Train Model V2 ---
# This uses the v2 compose file to train the challenger model with different hyperparameters.
Run-Compose-Stage -ComposeFile "docker-compose-v2.yml" -StageName "V2 Model Training"

# --- STAGE 3: Extract IDs and Run A/B Test ---
Write-Host "`n--- Preparing for Stage: A/B Test ---"

# Define paths to the metadata files created by the training stages
$v1_meta_path = "outputs/model_meta_Model_V1_Default.json"
$v2_meta_path = "outputs/model_meta_Model_V2_Low_Regularization.json"

# Check if the metadata files exist before trying to read them
if (-not (Test-Path $v1_meta_path)) {
    Write-Error "Could not find V1 metadata file at '$v1_meta_path'. The V1 training stage may have failed. Aborting."
    exit 1
}
if (-not (Test-Path $v2_meta_path)) {
    Write-Error "Could not find V2 metadata file at '$v2_meta_path'. The V2 training stage may have failed. Aborting."
    exit 1
}

# Read the JSON files and extract the MLflow Run IDs
$v1_meta = Get-Content -Raw -Path $v1_meta_path | ConvertFrom-Json
$v2_meta = Get-Content -Raw -Path $v2_meta_path | ConvertFrom-Json

$MODEL_V1_RUN_ID = $v1_meta.mlflow_run_id
$MODEL_V2_RUN_ID = $v2_meta.mlflow_run_id

# Check that the IDs were successfully extracted
if (-not $MODEL_V1_RUN_ID) {
    Write-Error "Failed to extract mlflow_run_id from '$v1_meta_path'. Aborting."
    exit 1
}
if (-not $MODEL_V2_RUN_ID) {
    Write-Error "Failed to extract mlflow_run_id from '$v2_meta_path'. Aborting."
    exit 1
}

Write-Host "Successfully extracted MLflow Run IDs:"
Write-Host "  - Model V1 Run ID: $MODEL_V1_RUN_ID"
Write-Host "  - Model V2 Run ID: $MODEL_V2_RUN_ID"

# Set the extracted IDs as environment variables for the Docker Compose command.
# In PowerShell, we set them in the environment block like this. They will be
# picked up by the `docker-compose` process.
$env:MODEL_V1_RUN_ID = $MODEL_V1_RUN_ID
$env:MODEL_V2_RUN_ID = $MODEL_V2_RUN_ID

# Run the final A/B test stage
Run-Compose-Stage -ComposeFile "docker-compose-ab-test.yml" -StageName "A/B Model Comparison"

Write-Host "`n==============================================="
Write-Host "--- Full Pipeline Completed Successfully! ---"
Write-Host "===============================================" 