#!/bin/bash
# Download git repo
if [ ! -d "keras_triplet_descriptor" ]; then
  git clone https://github.com/MatchLab-Imperial/keras_triplet_descriptor
fi
# Download hpatches data
if [ ! -d "hpatches" ]; then
  wget -O hpatches_data.zip https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip
  unzip -q ./hpatches_data.zip
  rm ./hpatches_data.zip
fi

# Update linux and install prerequisite libraries
apt update && apt install -y libsm6 libxext6 libxrender-dev

# Insall necessary python modules
pip install keras opencv-python
