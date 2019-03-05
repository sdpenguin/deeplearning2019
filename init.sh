#!/bin/bash
# Download keras_triplet_descriptor git repo
if [ ! -d "keras_triplet_descriptor" ]; then
  git clone https://github.com/MatchLab-Imperial/keras_triplet_descriptor
fi
# Make keras_triplet_descriptor recognised as a Python package
# and installs it
echo > ./keras_triplet_descriptor/__init__.py
# Replace the absolute package names with keras_triplet_descriptor.[Package] in utils
sed -i -e 's/read_data/keras_triplet_descriptor.read_data/g' ./keras_triplet_descriptor/utils.py
python ktd_setup.py install --force

# Download hpatches data
if [ ! -d "hpatches" ]; then
  wget -O hpatches_data.zip https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip
  unzip -q ./hpatches_data.zip
  rm ./hpatches_data.zip
fi

# Update linux and install prerequisite libraries
apt update && apt install -y libsm6 libxext6 libxrender-dev

# Insall necessary python modules
pip install keras opencv-python tqdm pandas matplotlib

# Install dl2019 Python package using the default python
/usr/bin/env python setup.py install --force