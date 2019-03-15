# deeplearning2019
The dl2019 Python package and Keras models for use in the Deep Learning 2019 coursework.

## About

This repository has three main components:

1. The dl2019 Python package which is a set of Python modules used to help run the models for the HPatches data.
2. Models impelemented within the package in ``dl2019/models/baseline.py``.
3. An ``init.sh`` script that runs the intial setup (downloading the required data and keras_triplet_descriptor as well as installing
keras_triplet_decriptor and dl2019 as Python packages so that they are accessible).

## How to Run

1. Download this git repository and run the ``init.sh`` file using ``bash``. This will download the HPatches dataset to the folder 
to your computer, as well as another repository by MatchLab-Imperial called ``keras_triplet_descriptor``. It will run two setup
files that install the current repository and the ``keras_triplet_descriptor`` repository as Python packages so that they are
globally accessible.
2. Run ``python dl2019/main.py`` with the required arguments. You can find out what is needed using the corresponding help ``python dl2019/main.py -h``.
3. To run tensorboard and visualise the training and model, run ``tensorboard --logdir log_dir`` where ``log_dir`` is the parameter specified when you run ``main.py``.

## Model Types

The current model types are listed below with their given architectures for both the Descriptor network and the Denoiser. To load the selected model type, set the parameter ``model_type`` equal to the model_type given when running the program:

| model_type | Denoiser       | Descriptor         |
| ---------- |:--------------:|:------------------:|
| baseline   | U-Net (ADAM LR:1e-5) | HardNet (L2 Net)  (ADAM LR:1e-3) |

In "Creating New Models" below, you can see how to add your own custom models to the program.

## Output directories

You can find the model weights and train/validation losses in the directory ``dump/model_type_[desc, denoise]``. Functions for loading and manipulating this data are already provided in the ``dl2019/eval/eval.py`` Python module.

### Plotting the Data

To load and plot the data, use the following code. You must change the given parameters to set the dir_dump directory and model_type and suffix to those you want to plot:

```
from dl2019.evaluate.evaluate import make_plot
from matplotlib import pyplot as plt

dir_dump = 'dump'
model_type = 'baseline'
suffix = 'desc'
suffix2 = None

make_plot(dir_dump, model_type, suffix, suffix2, max_epoch=10)
```

## Creating New Models

To create a new model, you need to do the following three things:

1. Create a class that implements your model in the file ``dl2019/models/baseline.py`` (or another Python module under models). Each class will be loaded by the main file and compiled as a model.

2. Pick a name for this model, and add it to the ``possible_models`` list in ``dl2019/utils/possibles.py``.

3. Add a condition to one or both of the loaders ``get_denoise_model`` and/or ``get_descriptor_model`` in ``dl2019/models/load.py``, depending on which ntework you implemented that will call the class you created. You can see existing examples in the functions already. Note that if you have created your class for the model in a separate module to ``baseline.py`` then you will need to import the module at the top of ``load.py``.

4. Run the program, with the option ``model_type`` equal to the option you created. Note that your model type name should not be the same as previously existing models, otherwise the program will think they are the same.

### Changing the Optmizer

The optimizer is currently can be seen and changed for the two different models in the compile methods of each model in ``dl2019/models/baseline.py``.

## Changing the dl2019 Code

To make modifications to the dl2019 Python package, and to make sure these modifications are reflected in the package, you need to rerun the ``setup.py`` script for ``dl2019``. Navigate to the top level directory of this repository and run ``python3 setup.py install --force``, in order to force overwrite the current package with the modified one.
