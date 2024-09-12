

# Create a screen session and run the Python file
echo "Starting controller"
screen -dmS controller bash -c "conda run -n llava --no-capture-output python -m llava.serve.controller --host 0.0.0.0 --port 10000; exec bash"

echo "Python script running in a screen session within the Conda environment llava."
