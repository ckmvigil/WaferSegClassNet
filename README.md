WaferSegClassNet - A Light-weight Network for Classification and Segmentation of Semiconductor Wafer Defects
==============================

This repository contains the source code of our paper, WaferSegClassNet (accepted for publication in <a href="https://www.sciencedirect.com/journal/computers-in-industry">Computers in Industry</a>).

we propose WaferSegClassNet (WSCN), a deep-learning model for simultaneously performing both classification and segmentation of defects on Wafer Maps. To the best of our knowledge, WSCN is the first wafer defect analysis model that performs both segmentation and classification. WSCN follows a multi-task learning framework to segment and classify an image simultaneously. 

<img src="reports/figures/wafermodelarchitecture_v2.png">


## Sample Results
<hr>

Check our <a href="check.com">project</a> page for more qualitative results.

<p align="center"><img src="reports/figures/Qualitative.png" width="840"></p>

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── config.py      <- All configuration params
    |   ├── util.py        <- All utilities functions
    │   │
    │   ├── data           <- Script to generate data in required format
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions and test performance.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   |   └── test_model.py
    |   |   └── network.py
    |   |   └── loss.py
    ├── inference          <- Inference directory, where predicted masks are stored.
    ├── logs               <- Logs directory for saving terminal output.
    ├── weights            <- Weights directory for saving checkpoints.
--------

## Get Started
<hr>
Dependencies:

```
pip install -r requirements.txt
```

### (Optional) Conda Environment Configuration

First, create a conda environment
```bash
conda create -n wscn # python=3
source activate wscn
```

Now, add dependencies

Now, you can install the required packages.
```bash
pip install -r requirements.txt
```

### Dataset

We have used MIXEDWM38 dataset which can be downloaded from <a href="https://github.com/Junliangwangdhu/WaferMap">here</a>. Download the dataset, unzip it and place ```Wafer_Map_Datasets.npz``` in ```data/raw/Wafer_Map_Datasets.npz```. 

To prepare the dataset ready for training, Run following command from ```/src``` directory.

```python data/make_dataset.py```

Above command should prepare Images, Labels, and Masks ready for training in ```data/processed``` directory.

### Training

change the hyperparameters and configuration parameters according to need in ```src/config.py```.

To train wscn, Run following command from ```/src``` directory.

```python models/train_model.py``` 

Above command will first pre-train encoder with N-Pair contrastive loss and then finetune segmentation and classification for given number of epochs.

### Prediction

To train wscn, Run following command from ```/src``` directory.

```python models/predict_model.py --image <path_of_an_image_in_numpy_format>``` 

Above command will predict the given image and save binary output mask in ```inference/``` directory.

### Test performance

To test wscn with trained model, Run following command from ```/src``` directory.

```python models/test_model.py ``` 

Above command will generate IOU Score, and DICE Score for segmentation output, and classification report and ROC AUC Curve for classification output.

## Citation

Yet to be updated

## License
<hr>
MIT License

