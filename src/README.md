## Source Code:

### Dataset Preparation:

To prepare the dataset ready for training, Run following command from ```/src``` directory.

```python data/make_dataset.py```

Above command should prepare Images, Labels, and Masks ready for training in ```data/processed``` directory.

### Training:
change the hyperparameters and configuration parameters according to need in ```src/config.py```.

To train wscn, Run following command from ```/src``` directory.

```python models/train_model.py```

Above command will first pre-train encoder with N-Pair contrastive loss and then finetune segmentation and classification for given number of epochs.

### Prediction:
To train wscn, Run following command from /src directory.

```python models/predict_model.py --image <path_of_an_image_in_numpy_format>```

Above command will predict the given image and save binary output mask in ``inference/``` directory.

### Test performance:
To test wscn with trained model, Run following command from ```/src``` directory.

```python models/test_model.py```

Above command will generate IOU Score, and DICE Score for segmentation output, and classification report and ROC AUC Curve for classification output.
