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

The current model types are listed below with their given architectures for both the Descriptor network and the Denoiser. To load the selected model type, set the parameter ``model_type_denoise`` or ``model_type_desc`` equal to the model_type given when running the program (if parameters are not specified for the optimisers, assume the [Keras defaults](https://keras.io/optimizers/)). The default optimiser listed in brackets is that that has been experimentally found to work best with the model and is the one loaded when you load the model. They may change in future if better optimiser parameters are found:

| model_type | Denoiser       | Descriptor         |
| ---------- |:--------------:|:------------------:|
| baseline   | Shallow U-Net (SGD LR:1e-5) | Modified HardNet (L2 Net)  (SGD LR:0.1) |
| baselinemse   | Shallow U-Net (SGD LR:1e-5) | N/A |
| unet       | U-Net (ADAM LR:1e-5) | N/A |

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

2. Pick a name for this model, and add it to the ``possible_denoise_models`` or ``possible_desc_models`` list in ``dl2019/utils/possibles.py``. This is necessary since otherwise the program will not recognise the model as valid when passing it as a command line argument (a preliminary check).

3. Add a condition to one or both of the loaders ``get_denoise_model`` and/o
r ``get_descriptor_model`` in ``dl2019/models/load.py``, depending on which ntework you implemented that will call the class you created. You can see existing examples in the functions already. Note that if you have created your class for the model in a separate module to ``baseline.py`` then you will need to import the module at the top of ``load.py``.

4. Run the program, with the option ``model_type`` equal to the option you created. Note that your model type name should not be the same as previously existing models, otherwise the program will think they are the same.

### Changing the Optmizer

The available optimizers can be appended to by adding a custom code to the ``dl2019/models/load_opt.py`` file and specifying this code for the ``optimizer`` flag (or in the Agenda file (see below)). The optimizer codes are given. Currently there are two dynamic optimizer codes: those for ``sgd`` and those for ``Adam``. The format is as follows ``[sgd, adam][lr](m[mom])`` where ``mom`` is the momentum and ``lr`` is the learning rate. Note that ``mom`` will be ignored for Adam.

## Running Multiple Models with a JSON Agenda

A powerful feature of this program is the ability to run mutliple models according to a JSON agenda. Example agendas are shown in the ``agendas`` folder, including many of the models with their corresponding optimizers included in the section Experiments and Results. Specify the list of models to run as a JSON list of JSON dictionaries, where each dictionary contains all the OPTIONAL parameters you would usually pass to the program to run it for one model. To specify the value ``None`` use the JSON equivalent ``null`` without quotes. JSON syntax can be easily learnt about online from a variety of sources and is very straightforward.

## Changing the dl2019 Code

To make modifications to the dl2019 Python package and to make sure these modifications are reflected in the package, you need to rerun the ``setup.py`` script for ``dl2019``. Navigate to the top level directory of this repository and run ``python3 setup.py install --force``, in order to force overwrite the current package with the modified one. In some cases, dependencies on ``keras_triplet_descriptor`` may not fall through correctly and you should also run the setup for ``keras_triplet_descriptor`` via ``python3 ktd_setup.py install --force``. Note that for this, you will need to have converted ``keras_triplet_descriptor`` to a Python package format - this is done by the ``init.py`` script automatically.

### Adding command line arguments

The following things MUST be checked off when you add a command line argument:

1. Add the argument to ``argparse.py``
2. Add the argument and its default value to the ``arg_list`` dictionary in ``possibles.py``
3. Add the line ``job['your_arg'] = parsed.your_arg`` to ``argparse.py``
4. (Optional): Add a sanity check for the argument to ``argparse.py``

The following may be done dependent on the purpose of the argument:

1. Change the ``main.py:main`` function to include your argument as a parameter
2. Pass the parameter to the ``main.py`` function, as you can see it being done for current arguments

## Folders

### HPatches Folder ``--dir_hpatches``

This is the folder where the hpatches data is extracted and obtained from when the program is run. ``init.sh`` should by default download it to ``./hpatches``.

### Dump Folder ``--dir_dump``

The dump folder contains several items:

1. Model folders with the ``.h5`` and ``.npy`` files from model runs. The naming scheme is ``[model_type]_[desc, denoise]_[optimizer]`` where ``optimizer`` and ``model_type`` are the same as the parameters passed to the program by the names ``--optimizer`` and ``--model-denoise`` or ``model-desc``.
2. The ``eval`` and ``results`` folders where the output of the benchmarking procedure will go, under the same subfolder names as above.
3. The saved denoiser data ``.dat`` files for faster loading so long as ``--nodisk`` is not set. 

### Keras Triplet Descriptor Folder ``--dir_ktd``

This is the folder containing the (modified if ``init.sh`` has done its job) ``keras_triplet_descriptor`` repository, with the original repository available publicly on Github, created by ``MatchLab-Imperial``.

## Experiments and Results

Results of the experiments that have been performed this far are given below. Note that some experiments were performed twice for different numbers of epochs. Some rows are empty since these experiments are yet to be performed. Note that the hyperparameters for the optimiser are also given since these may not be equal to those of the model in the code. This is because if at one stage the optimiser hyperparameters are the best known, they may not be in the future once more modifications to the optimiser have been explored for that particular model. Note that for consistency and comparison purposes the mean absolute error is given as the metric, regardless of the loss function used: 

### Denoiser

| model_type | Optimiser      | Epochs | Min Val Loss (MAE) |  id  |
| ---------- |:--------------:|:---------------------:|:------:|:---------------:|:------------:|:-------------:|:----:|
| baseline   | SGD LR:1e-5 Momentum:0.9 (Nesterov) | 100 | 5.184 | 1 |
| baseline   | Adam LR:1e-5   | 100 | 5.118 | 2 |
| baseline   | Adam LR:1e-5 Momentum:0.8 | 15 | 5.535  | 3 |
| baseline   | Adam LR:1e-3   | 15 | 5.016 | 4 |
| baseline  | Adam LR:1e-4   | 15 | 5.051 | 5 |
| unet   | SGD LR:1e-5 Momentum:0.9 |  |  | 6 |
| unet   | Adam LR:1e-5  |  |  | 7 |
| unet   | Adam LR:1e-4  | 10 |  | 8 |
| unet   | Adam LR:1e-3 | 10 |  | 9 |




### Descriptor

| model_type | Optimiser      | Denoiser Train (id) | Epochs | Min Val Loss (Triplet) | mAP Verification| mAP Matching | mAP Retrieval | Denoiser Eval (id) |
| ---------- |:--------------:|:-------------------:|:------:|:----------------------:|:---------------:|:------------:|:-------------:|:------------------:|
| baseline   | SGD LR:0.1 | 1 | 100 | 0.429 |  |  |  |  |
| baseline   | Adam LR:1e-3 | 2 | 100 | 0.444 | 0.770 | 0.156 | 0.463 | 2 |
| baseline   | SGD LR:0.1 | N/A | 100 | 0.047 |  |  |  |  |
| baseline   | Adam LR:1e-3 | N/A | 100 | 0.061 |  |  |  |  |
