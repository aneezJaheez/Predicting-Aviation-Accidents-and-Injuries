# Predicting-Aviation-Accidents-and-Injuries

## Index
* [Overview](#Overview)
* [Libraries Used](#Libraries-Used)
* [The Dataset](#The-Dataset)
  * [Dataset Source](#Data-Source)
  * [Cleaning the Dataset](#Cleaning-the-Dataset)
  * [Key Features](#Key-Features)
* [Predictions](#Predictions)
  * [Predicting Aircraft Damage](#Predicting-Aircraft-Damage-using-DTC)
  * [Predicting Injury Count](#Predicting-Injury-Count-using-Linear-Regression)
  * [Predicting Injury Severity](#Predicting-Injury-Severity-using-DCT)
* [Contributors](#Contributors)

## Overview
A machine learning project that aims to predict injury severity, count, and aircraft damage of aviation accidents by analyzing historical accidents with data points such as altitude, location, aircraft type, and passenger count.

## Libraries Used
* [Python 3.6.12](https://docs.python.org/3.6/)
* [scikit-learn 0.24.1](https://scikit-learn.org/stable/)
* [imbalanced-learn 0.8.0](https://imbalanced-learn.org/stable/index.html)
* [graphviz 0.16](https://graphviz.readthedocs.io/en/stable/manual.html)

and other essential libraries which include Pandas, Numpy, Matplotlib, and Seaborn.

## The Dataset

### Dataset Source

The data used in this project has been adapted from [Kaggle](https://www.kaggle.com/khsamaha/aviation-accident-database-synopses), a popular repository for community published datasets. The dataset contains records of air vehicles involved in aviation accidents. It also contains attributes holding information about the vehicle and other external factors that may have caused the accident.

### Cleaning the Dataset

Some of the steps in the data cleaning process are highlighted below:
1. Dropping attributes that are irrelevant to the goals of this project such as EventID, InvestigationType, EventDate, AirportCode etc.
2. Eliminating all amateur-built non-commercial air vehicles.
3. Mapping of inconsistent values for attributes such as aircraft model to a single consistent value for each model.
4. Eliminating records with no flight phase, location, or passenger data since this is crucial in our predictions.
5. Ensuring injury severity data matches the fatality data.
6. Introducing the approximate aircraft altitude as a new attribute estimated using the elevation of land and the aircrafts phase of flight.
7. Predicting missing weather condition data using existing attributes with a decision tree classifier.

For more information about how the cleanup was carried out and the reasoning behind each step, check out our [notebook](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Notebooks/Data_Extraction_Cleanup.ipynb)

### Key Features

Some of the key features of the dataset inferred during the exploratory analysis is as follows:
1. Shape of the dataset after cleaning : 12954 records × 15 attributes, of which 7 attributes are categorical
2. The dataset was heavily skewed, some examples of which include:
  1. Weather condition - VMC:IMC :: 13:1
  1. Injury Severity - Non-Fatal:Fatal:Incident :: 24:5:1
  1. Aircraft Damage - Substantial:Destroyed:Minor :: 25:4:1
3. Most accidents took place during landing
4. The injury severity has a strong correlation with the aircraft damage.
5. A negative correlation between the weather conditions and aircraft damage was observed.

For further insigts about the dataset and detailed visualizations, check out the [exploratory analysis notebook](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Notebooks/Data_Extraction_Cleanup.ipynb)
  

## Contributors
* Aneez Ahmed Jaheezuddin
* [You Zhi Min](https://github.com/zzzhimin)
* [Yeong Wei Xian](https://github.com/wxiannnn)
* [Kenneth Lee](https://github.com/klee046)
