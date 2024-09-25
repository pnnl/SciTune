# Define variables
cd "${0%/*}"
CONDA_PATH="/home/ubuntu/miniconda3/bin/conda"  # Replace this with your Conda path
ENV_NAME="llava"         # Replace this with your Conda environment name


# Activate Conda environment
# Initialize Conda in the script
eval "$(conda shell.bash hook)"


conda activate "$ENV_NAME"

# Create a screen session and run the Python file
screen -dmS modelWorker2 bash -c "python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model-path liuhaotian/llava-v1.5-13b --load-4bit; exec bash"

echo "Model Worker running in a screen session modelWorker2 within the Conda environment '$ENV_NAME'."
