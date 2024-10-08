#!/bin/bash
cd ../dataset

# Download Arxiv data
echo -e "\n\n*****Downloading ArxivCap*****"
git clone https://huggingface.co/datasets/MMInstruction/ArxivCap
echo -e "\n\n*****Downloading Scicap*****"
git clone https://huggingface.co/datasets/CrowdAILab/scicap
echo -e "\n\n*****Downloading ScienceQA*****"
git clone https://huggingface.co/datasets/derek-thomas/ScienceQA
echo -e "\n\n*****Downloading MathVista*****"
git clone https://huggingface.co/datasets/AI4Math/MathVista


echo -e "\n\n*****Downloading VisText*****"
function usage {
  echo "usage: $0 [--images] [--scenegraphs] [--image_guided]"
  echo "  --images          Download rasterized chart images."
  echo "  --scenegraphs     Download full scenegraphs."
  echo "  --vl_spec         Download Vega-Lite specs."
  echo "  --image_guided    Download multimodal features and weights."
  exit 1
}

# Default parameter values.
images=false        # true to download images; false otherwise.
scenegraphs=false   # true to download scenegraphs; false otherwise.
vl_spec=false       # true to download vega-lite specs; false otherwise.
mm=false            # true to download multimodal features and weights; false otherwise.


# Update parameters based on arguments passed to the script.
while [[ $1 != "" ]]; do
    case $1 in
    --images)
        images=true
        ;;
    --scenegraphs)
        scenegraphs=true
        ;;
    --vl_spec)
        vl_spec=true
        ;;
    --image_guided)
        mm=true
    esac
    shift
done

# Download tabular data
echo "Downloading tabular data zip"
wget https://vis.csail.mit.edu/vistext/tabular.zip -P ./data/

# Download images
if [[ $images = true ]]; then
    echo "Downloading rasterized chart images zip."
    wget https://vis.csail.mit.edu/vistext/images.zip -P ./data/
fi
# Download scenegraphs
if [[ $scenegraphs = true ]]; then
    echo "Downloading full scenegraphs zip."
    wget https://vis.csail.mit.edu/vistext/scenegraphs.zip -P ./data/
fi
# Download vl_specs
if [[ $vl_spec = true ]]; then
    echo "Downloading full vega-lite specs zip."
    wget https://vis.csail.mit.edu/vistext/vl_spec.zip -P ./data/
fi
# Download multimodal features
if [[ $mm = true ]]; then
    echo "Downloading multimodal features and weights zips."
    wget https://vis.csail.mit.edu/vistext/visual_features.zip -P ./data/
    wget https://vis.csail.mit.edu/vistext/vl_weights.zip -P ./models/
fi

echo "Downloading complete. Unzipping archives."

# Unzip tabular data
unzip ./data/tabular.zip -d ./data/
# Unzip images
if [[ $images = true ]]; then
    unzip ./data/images.zip -d ./data/
fi
# Unzip scenegraphs
if [[ $scenegraphs = true ]]; then
    unzip ./data/scenegraphs.zip -d ./data/
fi
# Unzip vl_spec
if [[ $vl_spec = true ]]; then
    unzip ./data/vl_spec.zip -d ./data/
fi
# Unzip multimodal features
if [[ $mm = true ]]; then
    unzip ./data/visual_features.zip -d ./data/
    unzip ./models/vl_weights.zip -d ./models/
fi

mv ArxivCap/ arxivcap/
mv data/ vistext/
mv ScienceQA/ scienceqa/
mv MathVista/ mathvista/