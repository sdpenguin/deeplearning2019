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
globally accessible. Note that if you do not wish to run ``init.sh`` directly then you may wish to have a look at the section ``Setting Up the Environment`` below to understand what the script does so that you can do the equivalent operations yourself.
2. Run ``python dl2019/main.py`` with the required arguments. You can find out what is needed using the corresponding help ``python dl2019/main.py -h``.
3. To run tensorboard and visualise the training and model, run ``tensorboard --logdir log_dir`` where ``log_dir`` is the parameter specified when you run ``main.py``.

## Setting up the Environment (init.sh)

1. Download the ``keras_triplet_descriptor`` repository from Github: https://github.com/MatchLab-Imperial/keras_triplet_descriptor by running ``git clone https://github.com/MatchLab-Imperial/keras_triplet_descriptor``
3. Run the following two lines in order to make ``keras_triplet_descriptor`` an installable package and install it so that it is globally accessible:
```
echo > ./keras_triplet_descriptor/__init__.py
# Replace the absolute package names with keras_triplet_descriptor.[Package] in utils
sed -i -e 's/ read_data/ keras_triplet_descriptor.read_data/g' ./keras_triplet_descriptor/utils.py
python3 ktd_setup.py install --force
```
4. Run the following lines of code to download the dataset into a folder called ``hpatches``:
```
wget -O hpatches_data.zip https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip
  unzip -q ./hpatches_data.zip
  rm ./hpatches_data.zip
```
5. Install the following Python modules using ``pip``. Note that ``tensorflow`` is reinstalled. You may skip this step if you do not wish to reinstall ``tensorflow``. You will be prompted as to whether you do if you run ``init.sh`` in any case:
```
pip install keras opencv-python tqdm pandas matplotlib joblib dill tabulate scikit-image
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-gpu
```
6. Run the setup script via ``python3 setup.py install --force`` to install the ``dl2019`` Python package.

## Running the program

There are currently two ways to run the program. Either you run the program for one set of parameters by specifying the command line arguments directly, or you can use the ``-f`` flag to specify the path to a ``.json`` file containing a list of dictionaries, where each dictionary is a dictionary of the optional arguments. An example ``.json`` file containing a selection of the model specifications run is shown in ``agendas/models.json``.

### Running with command line arguments

You can run a single denoiser and descriptor model using the following syntax:

``python dl2019/main.py dir_hpatches dir_dump dir_ktd [ -e --evaluate ] [ --pca ] [ --model-denoise baseline ] [ --model-desc baseline ] [ --epochs-denoise 0 ] [ --epochs-desc 0 ] [ --nodisk ] [ --use-clean ] [ --denoise-suffix ] [ --denoiser-train ] [ --optimizer-desc default ] [ --optimizer-denoise default ] [ --keep-results ]

Here is an explanation behind the parameters, which can also be obtained by running ``python dl2019/main.py --help``. When these command line arguments are translated into parameter names the dashes in their names are replaced by underscores e.g. ``dir-hpatches`` becomes ``dir_hpatches``:

| parameter | Default Value | Meaning |
| ---------- |:--------------:|:------------------:|
| dir-hpatches | N/A | Base hpatches directory containing all hpatches data in the default format. |
| dir-dump | N/A | Directory to place DenoiseHPatchesImproved formatted hpatches data, weights and losses. |
| dir-ktd | N/A | The keras_triplet_descriptor respository directory. |
| agenda-file (-f) | False | Specify a file path containing a JSON specification for jobs to run. Please see README for spec details. |
| evaluate | False | Set this flag to run evaluation (verification/matching/retrieval) tests on the specified descriptor model (with or without denoiser depending on use-clean. |
| pca | default=False | Use the pca_power_law for evaluation. This may take longer, but generates more information. |
| model-denoise | baseline | The model to run for the denoiser. Must be one of the possible_denoise_models in utils/possibles.py. |
| model-desc | baseline | The model to run for the descriptor. Must be one of possible_desc_models in utils/possibles.py |
| epochs-denoise | 0 | Number of epochs for the denoiser to be run |
epochs-desc | 0 | Number of epochs for the descriptor |
| optimizer-desc | default | Descriptor optimizer code to specify the optimizers you want to use. Default will be loaded if not specified for the model |
| optimizer-denoise | baseline | Denoiser optimizer code to specify the optimizers you want to use. Default will be loaded if not specified for the model. For more information on the possible optimizer codes, see below. |
| nodisk | False | Set this flag to avoid saving or loading HPatches Denoiser Generators from disk. The HPatches data will be regenerated from scratch and not saved after it is generated. This may be useful for low-RAM systems. |
| use-clean | False | Set this flag to train/evaluate the descriptor model on clean data, instead of data denoised using the denoiser. |
| denoise-suffix | None | Optional suffix for the denoiser folder. |
| denoisertrain | None | Suffix specifying which denoiser the descriptor should be trained on data denoised by. If none specified then the model parameters are used to deduce the denoiser. If the model parameters do not correspond to the denoiser model given, then the descriptor WILL NOT be trained FURTHER, but may be evaulated on the denoiser parameters. |

#### Optimizer Codes

Please see the section on Changing the Optmizer below.

### Running with a json script

Run the following command:

``python dl2019/main.py dir_hpatches dir_dump dir_ktd -f agendas/models.json``

Where you replace ``models.json`` with your own specification. Any optional paramter can be specified in a job definition in the ``json`` file (any parameter other than dir_hpatches dir_dump, dir_ktd) as well as of course the ``-f`` parameter. Note that if the ``-f`` option is used, then the other optional parameters are ignored.

### A note on denoisertrain

The denoisertrain, also known in ``main.py`` as ``desc_suffix`` for legacy reasons, will be one of three things:

1. Of the following format (model_denoise)_(optimizer_denoise) e.g. ``baseline_adam1e-3m0.9``.
2. ``none_default``: this means that the descriptor data was trained on noisy data, but no denoiser was used to denoise the data prior to passing it to the descriptor.
3. ``clean``: the descriptor was trained on clean data.

## Overview of the ``dl2019`` package

The ``dl2019`` has grown to be a complex set of interacting scripts, however there is a method to the madness. The first port of call for anything is the script ``main.py``. This is the script that calls all other modules and scripts in the package and is the entry point of the program. It contains top level functions that will load models, load generators and run the models, while handling the necessary parameters.

The ``dl2019`` package contains three subpackages: ``utils``, ``models`` and ``evaluate``.They respectively perform general operations, model operations and evaluation (testing) operations.

The second important script is ``utils/argparse.py``. This is a script that includes the parsing of the command line arguments. If you wish to modify these, then this is the script you must look to first. Note that in order to modify the command line arguments, you must also change a few other things - a detailed explanation is given in the section ``Adding command line arguments``.

The way that ``main.py`` works can be seen by examining the comments in the file.

## Model Types

The current model types are listed below with their given architectures for both the Descriptor network and the Denoiser. To load the selected model type, set the parameter ``model_type_denoise`` or ``model_type_desc`` equal to the model_type given when running the program (if parameters are not specified for the optimisers, assume the [Keras defaults](https://keras.io/optimizers/)). The default optimiser listed in brackets is that that has been experimentally found to work best with the model and is the one loaded when you load the model. They may change in future if better optimiser parameters are found:

| model_type | Denoiser       | Descriptor         |
| ---------- |:--------------:|:------------------:|
| baseline   | Shallow U-Net (SGD LR:1e-5) | Modified HardNet (L2 Net)  (SGD LR:0.1) |
| baselinemse   | Shallow U-Net (SGD LR:1e-5) | N/A |
| unet       | Modified U-Net (ADAM LR:1e-5) | N/A |
| dncnn       | 17 Layer DnCNN (ADAM LR:1e-5) | N/A |
| baselinedog       | N/A | Baseline Architecture with Difference of Gaussians Preliminary Filtering (ADAM LR:1e-5) |
| baseline(100,250,500)       | N/A | Baseline Architecture with Batch Sizes of 100, 250 or 500 (pick one) (ADAM LR:1e-5) |

In "Creating New Models" below, you can see how to add your own custom models to the program.

## Output directories

You can find the model weights and train/validation losses in the directory ``dump/model_type_[desc, denoise]``. Functions for loading and manipulating this data are already provided in the ``dl2019/evaluate/evaluate.py`` Python module.

### Plotting the Data

To load and plot the data, use the following code. You must change the given parameters to set the dir_dump directory and model_type and suffix to those you want to plot:

```
from dl2019.evaluate.evaluate import make_plot
from matplotlib import pyplot as plt

dir_dump = 'dump'
model_type = 'baseline'
suffix = 'desc'
optimizer_desc = 'sgd1e-1'
suffix2 = 'default_none_baseline_sgd1e-4' # The denoisers used for evaluation and testing respectively

# You can set max_epoch to only plot up to a particular epoch, or override_checks to skip validation checks on the model_type, suffix and optimizer
make_plot(dir_dump, model_type, suffix, optimizer, suffix2=None, max_epoch=20, override_checks=False, mae=True)
```

Note: suffix2 should be of the form: ``(model_denoise_train)_(optimizer_denoise_train)_(model_denoise_eval)_(optimizer_denoise_eval)``. Please see the note on ``denoisertrain`` above. ``denoisertrain`` is equivalent to ``(model_denoise_train)_(optimizer_denoise_train)``.

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

1. Model folders with the ``.h5`` and ``.npy`` files from model runs. The naming scheme is ``[model_type]_[desc, denoise]_[optimizer]_[suffix]`` where ``optimizer`` and ``model_type`` are the same as the parameters passed to the program by the names ``--optimizer-denoise`` or ``optimizer-desc`` and ``--model-denoise`` or ``model-desc``. Suffix will be the denoiser used for training of the format ``(model_denoise)_(optimizer_denoise)`` for the descriptor, and will either be nothing or the value passed to ``suffix-denoise`` (see above section on command line arguments) for the denoiser.
2. The ``eval`` and ``results`` folders where the output of the benchmarking procedure will go, under the same subfolder names as above.
3. The saved denoiser data ``.dat`` files for faster loading so long as ``--nodisk`` is not set.

### Keras Triplet Descriptor Folder ``--dir_ktd``

This is the folder containing the (modified if ``init.sh`` has done its job) ``keras_triplet_descriptor`` repository, with the original repository available publicly on Github, created by ``MatchLab-Imperial``.

## Experiments and Results

A few of the experiments that have been performed this far are given below. Note that some experiments were performed twice for different numbers of epochs. Some rows are empty since these experiments are yet to be performed. Note that the hyperparameters for the optimiser are also given since these may not be equal to those of the model in the code. This is because if at one stage the optimiser hyperparameters are the best known, they may not be in the future once more modifications to the optimiser have been explored for that particular model. Note that for consistency and comparison purposes the mean absolute error is given as the metric, regardless of the loss function used: 

### Baseline Denoiser MAE Results

| model_type | Optimiser      | Epochs | Min Val Loss (MAE) |  id  |
| ---------- |:--------------:|:---------------------:|:------:|:---------------:|
| baseline   | SGD LR:1e-5 Momentum:0.9 (Nesterov) | 100 | 5.184 | 1 |
| baseline   | Adam LR:1e-5   | 100 | 5.118 | 2 |
| baseline   | Adam LR:1e-5 Momentum:0.8 | 15 | 5.535  | 3 |
| baseline   | Adam LR:1e-3   | 15 | 5.016 | 4 |
| baseline   | Adam LR:1e-4   | 15 | 5.051 | 5 |



### Evaluation Results

Note the optimiser is specified as an optimizer code whose form is given in the section on Changing the Optimizer above.

| model_type_denoise | optimizer_denoise | model_type_desc | optimizer_desc | mAP Verification| mAP Matching | mAP Retrieval |
| ---------- |:--------------:|:-------------------:|:------:|:----------------------:|:---------------:|:------------:|
| baseline | Adam1e-3 | baseline | SGD1e-1 | 0.809 | 0.213 | 0.510 |
| baseline | Adam1e-4 | baseline | SGD1e-1 | 0.799 | 0.196 | 0.490 |
| baseline | Adam1e5m0.99 | baseline | SGD1e-1 | 0.817 | 0.221 | 0.525 |
| dncnn | Adam1e-3 | baseline | SGD1e-1 | 0.802 | 0.207 | 0.500 |
| unet | Adam1e-3 | baseline | SGD1e-1 | 0.800 | 0.202 | 0.493 |
| unet | Adam1e-4 | baseline | SGD1e-1 | 0.801 | 0.205 | 0.500 |
| baseline | Adam1e-4 | baseline100 | SGD1e-1 | 0.817 | 0.221 | 0.525 |
| baseline | Adam1e-4 | baselinedog | Adam1e-4 |0.827 | 0.245 | 0.519 |

#### Conclusions

We see that the DoG method is the one that proved to show the best performance in both the verification and matching tasks. It would be interesting in future to run the DoG method again, using the optimized denoisers as the denoisers.

The odd thing about the performance is that it seems that the optimizers with the lower MAE (the baseline ones) outperformed the U-Net and DnCNN optimizers in the three tasks given the same . This may be because the descriptors trained on this data had an easier job to do presented with cleaner data, therefore despite the MAE going down during training, during testing, the descriptor may not have been able to perform as well on novel data.

Any of the experiments above can be rerun as follows. Note that if you have used ``init.sh`` to setup your directory, then dir_hpatches=``hpatches``, dir_dump=``dump`` and dir_ktd=``keras_triplet_descriptor``:

``python3 dl2019/main.py dir_hpatches dir_dump dir_ktd -e --model-denoise model_type_denoise --epochs-denoise 20 --optimizer-denoise optimizer_denoise --model-desc model_type_desc --epochs-desc 20 --optimizer-desc optimizer_desc``
