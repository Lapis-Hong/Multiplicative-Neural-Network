# Multiplicative Neural Network (MNN)

## Content
* [Overview](#overview)
* [Dataset](#dataset)
* [Model](#model)
* [Usage](#usage)
* [Experiments](#experiments)


## Overview
Here, we develop a brandly new framework, called **MNN** for general structural data classification tasks, such as CTR prediction, recommend system, etc.
The code is based on TensorFlow high level `tf.estimator.Estimator` API. 
We use Kaggle Criteo and Avazu Dataset as examples.


### Requirements
- Python 3.6
- TensorFlow >= 1.10
- NumPy
- pyyaml

## Dataset

### 1. Criteo
Kaggle Criteo Dataset [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)

#### Data descriptions
- train.csv - The training set consists of a portion of Criteo's traffic over a period of 7 days. Each row corresponds to a display ad served by Criteo. Positive (clicked) and negatives (non-clicked) examples have both been subsampled at different rates in order to reduce the dataset size. The examples are chronologically ordered.
- test.csv - The test set is computed in the same way as the training set but for events on the day following the training period.
Note: the test.csv file label is unreleased, here we randomly split train.csv into train, dev, test set.

#### Data fields
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 

The semantic of the features is undisclosed.
When a value is missing, the field is empty.

### 2. Avazu
Kaggle Avazu Dataset [Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)

#### Data descriptions
- train - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
- test - Test set. 1 day of ads to for testing your model predictions. 
Note: the test file label is unreleased, here we randomly split train.csv into train, dev, test set.

#### Data fields
- id: ad identifier
- click: 0/1 for non-click/click
- hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
- C1 -- anonymized categorical variable
- banner_pos
- site_id
- site_domain
- site_category
- app_id
- app_domain
- app_category
- device_id
- device_ip
- device_model
- device_type
- device_conn_type
- C14-C21 -- anonymized categorical variables

## Model


## Usage
### Setup
```
cd conf
vim feature.yaml
vim train.yaml
...
```
### Training
You can run the code locally as follows:
```
python train.py
```
### Testing
```
python test.py
```
### TensorBoard
Run TensorBoard to inspect the details about the graph and training progression.
```
tensorboard --logdir=./model/mnn
```

## Experiments
### settings
For continuous features, we use log transform as input,
for category features, we set hash_bucket_size according to its values size, and embeded into dense representations.
The specific parameters setting see `conf/*/train.yaml`

### criteo dataset

### avazu dataset


