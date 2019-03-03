#!/bin/bash
# Download git repo
git clone https://github.com/MatchLab-Imperial/keras_triplet_descriptor
# Download hpatches data
wget -O hpatches_data.zip https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip
unzip -q ./hpatches_data.zip
rm ./hpatches_data.zip

# Insall necessary python modules
pip install keras opencv-python
