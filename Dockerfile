# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for Rust and some Python packages
# Switch to root to install packages
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo using rustup
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal

# Switch back to default user (optional, depends on base image, often 'app' or non-root)
# If the base image doesn't define a non-root user, you might skip this or create one.
# USER app # Or whatever user the base image provides if not root

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
# This command will now have access to Rust/Cargo
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code files into the container at /app
# Ensure all necessary scripts are copied
COPY pipeline.py .
COPY data_quality_tests.py .
# If you have other local modules imported by your scripts, copy them too
# COPY other_module.py .

# --- IMPORTANT ---
# The data file (accepted_2007_to_2018Q4.csv) is likely large and should NOT be copied here.
# It should be mounted as a volume when running the container.
# Similarly, the mlruns directory should be mounted to persist MLflow data.

# Define the command to run your application
# This will execute when the container starts
CMD ["python", "pipeline.py"] 