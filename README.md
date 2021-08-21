# Covid-19 Detection via X-Ray Images

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Data](#data)
  * [Frameworks used](#frameworks-used)
  * [Labels](#labels)
  * [Evaluation Metric](#evaluation-metric)
  * [Cross-Validation](#cross-validation)
  * [Data-Augmentations](#data-augmentations)
  * [Result](#result)

## Overview
This is a simple image classification Project trained on the top of Pytorch API. The trained model that takes X-Ray image as an input and predict the class of image from {'Malignant', 'Benign'}
This is a Image Classification Project trained on the top of Pytorch API. The task is to identify COVID-19 abnormalities on chest radiographs. In particular, the task categorize the radiographs as negative for pneumonia or typical, indeterminate, or atypical for COVID-19. The model works with Imaging data and annotations from a group of radiologists.

## Motivation
Five times more deadly than the flu, COVID-19 causes significant morbidity and mortality. Like other pneumonias, pulmonary infection with COVID-19 results in inflammation and fluid in the lungs. COVID-19 looks very similar to other viral and bacterial pneumonias on chest radiographs, which makes it difficult to diagnose. This computer vision model to detect COVID-19 would help doctors provide a quick and confident diagnosis. As a result, patients could get the right treatment before the most severe effects of the virus take hold.

## Data
The train dataset comprises 6,334 chest scans in JPEG format, which were de-identified to protect patient privacy. All images were labeled by a panel of experienced radiologists for the presence of opacities as well as overall appearance.

To download the data you can go here [dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification/data) or use the kaggle api by typing the following in your terminal ```kaggle competitions download -c siim-covid19-detection```

The data can be found in the folder (`input/train.csv`)

The Images were not provided here due to the large size.

## Frameworks-used
1. Pillow
2. OpenCV
3. Pandas
4. scikit-learn
5. Pytorch
6. Albumentations


![](https://forthebadge.com/images/badges/made-with-python.svg)<br>
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/02/pytorch-logo-flat-300x210.png) 

## Labels
{0: 'Negative for Pneumonia',
 1: 'Typical Appearance',
 2: 'Indeterminate Appearance',
 3: 'Atypical Appearance'}
 
 ## Evaluation Metric

The Dataset was particularly skewed.

Thus Accuracy is not a good metric here, We are going to Validate our model using "AUC ROC"

## Cross-Validation
We are going to use Stratified K Fold cross-validation method, because there is high class imbalance in the dataset. We are using 5 folds.

## Model

We are using CNN based model with the efficien-net-b7 architecture for the classification of the leafs.

## Data-Augmentations
**train dataset transformations**
        1. RandomResizedCrop
        2. Transpose
        3. HorizontalFlip
        4.VerticalFlip
        5. ShiftScaleRotate
        6. HueSaturationValue
        7. RandomBrightnessContrast
        8. Normalize
        9. CoarseDropout
        10. Cutout

**valid dataset transformations**
        1. CenterCrop
        2. Resize
        3. Normalize

## Result

* We get an Cross Validation AUC-ROC Score of 0.722 
