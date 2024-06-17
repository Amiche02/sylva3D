#!/bin/bash

# Define the model URLs
MODEL_1_URL="https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"
MODEL_2_URL="https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt"

# Define the model file names
MODEL_1_NAME="dpt_beit_large_512.pt"
MODEL_2_NAME="dpt_swin2_large_384.pt"

# Path to the weights directory
PROJECT_DIR=$(pwd)
WEIGHTS_DIR="${PROJECT_DIR}/checkpoints/weights"

# Create the weights directory if it does not exist
mkdir -p ${WEIGHTS_DIR}

# Display options to the user
echo "Please choose the model to download:"
echo "1) dpt_beit_large_512 - Highest quality"
echo "2) dpt_swin2_large_384 - Good quality with better speed-performance trade-off"
read -p "Enter the number of your choice: " CHOICE

# Download the chosen model
case $CHOICE in
    1)
        echo "Downloading dpt_beit_large_512..."
        wget -O ${WEIGHTS_DIR}/${MODEL_1_NAME} ${MODEL_1_URL}
        echo "Model downloaded to ${WEIGHTS_DIR}/${MODEL_1_NAME}"
        ;;
    2)
        echo "Downloading dpt_swin2_large_384..."
        wget -O ${WEIGHTS_DIR}/${MODEL_2_NAME} ${MODEL_2_URL}
        echo "Model downloaded to ${WEIGHTS_DIR}/${MODEL_2_NAME}"
        ;;
    *)
        echo "Invalid choice. Please run the script again and enter 1 or 2."
        ;;
esac
