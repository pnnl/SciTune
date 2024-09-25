# Define variables
#cd "${0%/*}"
#CONDA_PATH="/home/ubuntu/miniconda3/bin/conda"  # Replace this with your Conda path
#ENV_NAME="llava"         # Replace this with your Conda environment name


# Activate Conda environment
# Initialize Conda in the script
#eval "$(conda shell.bash hook)"

DATA_DIR=${DATA_DIR:-/tmp/data}
echo "Running modelworker data dir - $DATA_DIR"
# # Create a screen session and run the Python file
# screen -dmS modelWorker bash -c "python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ~/scitune_data/models/LLAVA/llava-13B --load-4bit; exec bash"

screen -dmS modelWorker bash -c "conda run -n llava --no-capture-output python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path $DATA_DIR/scitune_data/models/scienceqa/scitune-llava-checkpoint-4500 --load-4bit; exec bash"

# screen -dmS modelWorker bash -c "python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b --load-4bit; exec bash"

echo "Model Worker running in a screen session modelWorker within the Conda environment llava."