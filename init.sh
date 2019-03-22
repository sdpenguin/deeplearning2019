#!/bin/bash
# Possibly install necessary programs for containers
apt-get install wget
apt-get install zip
apt-get install git
# Download keras_triplet_descriptor git repo
if [ ! -d "keras_triplet_descriptor" ]; then
  git clone https://github.com/MatchLab-Imperial/keras_triplet_descriptor
fi
# Make keras_triplet_descriptor recognised as a Python package
# and installs it
echo > ./keras_triplet_descriptor/__init__.py
# Replace the absolute package names with keras_triplet_descriptor.[Package] in utils
sed -i -e 's/ read_data/ keras_triplet_descriptor.read_data/g' ./keras_triplet_descriptor/utils.py
# The below additional modficiatoins now been substituted by moving keras_triplet_descriptor.hpatches-benchmark to dl2019.evaluate.hpatches_benchmark
#cp "./keras_triplet_descriptor/hpatches-benchmark" ./keras_triplet_descriptor/hpatches_benchmark
#sed -i -e 's/ utils/ keras_triplet_descriptor.hpatches_benchmark.utils/g' ./keras_triplet_descriptor/hpatches_benchmark/utils/tasks.py
#sed -i -e 's/ utils/ keras_triplet_descriptor.hpatches_benchmark.utils/g' ./keras_triplet_descriptor/hpatches_benchmark/utils/results.py
python3 ktd_setup.py install --force

# Download hpatches data
if [ ! -d "hpatches" ]; then
  wget -O hpatches_data.zip https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip
  unzip -q ./hpatches_data.zip
  rm ./hpatches_data.zip
fi

# Update linux and install prerequisite libraries
apt update && apt install -y libsm6 libxext6 libxrender-dev

# Insall necessary python modules
pip install keras opencv-python tqdm pandas matplotlib joblib dill tabulate scikit-image
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-gpu

# Install dl2019 Python package using the default python
/usr/bin/env python3 setup.py install --force
