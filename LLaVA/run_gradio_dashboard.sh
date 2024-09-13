
echo "running dashboard"
# Create a screen session and run the Python file
screen -dmS dashboard bash -c "conda run -n llava --no-capture-output python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload; exec bash"

echo "Python script running in a screen session within the Conda environment llava."
