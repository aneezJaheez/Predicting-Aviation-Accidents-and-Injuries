# Predicting-Aviation-Accidents-and-Injuries

## Index
* [Overview](#Overview)
* [Libraries Used](#Libraries-Used)
* [The Dataset](#The-Dataset)
  * [Dataset Source](#Dataset-Source)
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

![Dataset Attribute Definitions](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/Dataset%20Attributes.png)

### Cleaning the Dataset

Some of the steps in the data cleaning process are highlighted below:
1. Dropping attributes that are irrelevant to the goals of this project such as EventID, InvestigationType, EventDate, AirportCode etc.
2. Eliminating all amateur-built non-commercial air vehicles.
3. Mapping of inconsistent values for attributes such as aircraft model to a single consistent value for each model.
4. Eliminating records with no flight phase, location, or passenger data since this is crucial in our predictions.
5. Ensuring injury severity data matches the fatality data.
6. Introducing the approximate aircraft altitude as a new attribute estimated using the elevation of land and the aircrafts phase of flight.

```python
for index, row in aviationData.iterrows():
    if(aviationData.at[index, 'Broad.Phase.of.Flight'] == 'CRUISE' or aviationData.at[index, 'Broad.Phase.of.Flight'] == 'MANEUVERING' or aviationData.at[index, 'Broad.Phase.of.Flight'] == 'GO-AROUND'):
        if(aviationData.at[index, 'Purpose.of.Flight'] == 'PERSONAL'):
            altitude = 12496.8 - aviationData.at[index, 'Altitude']
            if(altitude > 0):
                aviationData.at[index, 'Altitude'] = altitude
        else:
            altitude = 10972 - aviationData.at[index, 'Altitude']
            if(altitude > 0):
                aviationData.at[index, 'Altitude'] = altitude
    
    if(aviationData.at[index, 'Broad.Phase.of.Flight'] == 'APPROACH' or aviationData.at[index, 'Broad.Phase.of.Flight'] == 'DESCENT' or aviationData.at[index, 'Broad.Phase.of.Flight'] == 'CLIMB'):
        altitude = 700 - aviationData.at[index, 'Altitude']
        if(altitude > 0):
            aviationData.at[index, 'Altitude'] = altitude
```

8. Predicting missing weather condition data using existing attributes with a decision tree classifier.

For more information about how the cleanup was carried out and the reasoning behind each step, check out our [notebook](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Notebooks/Data_Extraction_Cleanup.ipynb)

### Key Features

Some of the key features of the dataset inferred during the exploratory analysis is as follows:
<ol>
 <li>Shape of the dataset after cleaning : 12954 records × 15 attributes, of which 7 attributes are categorical</li>

![Categorical attributes](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/catData.png)

![Numerical attributes](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/numData.png)

![Numerical Attributes summary](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/numDataSummary.png)

 <li>The dataset was heavily skewed, some examples of which include:</li>
 <ol>
  <li>Weather condition - VMC:IMC :: 13:1</li>
  <li>Injury Severity - Non-Fatal:Fatal:Incident :: 24:5:1</li>
  <li>Aircraft Damage - Substantial:Destroyed:Minor :: 25:4:1</li>

![Weather Condition Percentages](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/skewed.png)

 </ol>
 <li>Most accidents took place during landing.</li>

![Phase of Flight](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/PhaseOfFlight.png)

 <li>The injury severity has a strong correlation with the aircraft damage.</li>

![Injury Severity and Aircraft Damage](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/Injury%20Severity%20Aircraft%20Damage.png)

 <li>A negative correlation between the weather conditions and aircraft damage was observed.</li>

![Weather Condition and Aircraft Damage](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/Weather%20Condition%20and%20Aircraft%20Damage.png)

</ol>

For further insigts about the dataset and detailed visualizations, check out the [exploratory analysis notebook](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Notebooks/Data_Extraction_Cleanup.ipynb)
  

## Contributors
* Aneez Ahmed Jaheezuddin
* [You Zhi Min](https://github.com/zzzhimin)
* [Yeong Wei Xian](https://github.com/wxiannnn)
* [Kenneth Lee](https://github.com/klee046)
