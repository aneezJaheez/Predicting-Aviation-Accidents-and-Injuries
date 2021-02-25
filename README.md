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
  * [Predicting Injury Severity](#Predicting-Injury-Severity-using-DTC)
* [The Team](#The-Team)

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
 <li>Shape of the dataset after cleaning : 12954 records × 15 attributes, of which 7 attributes are categorical</li><br/>

![Categorical attributes](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/catData.png)

![Numerical attributes](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/numData.png)

![Numerical Attributes summary](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/numDataSummary.png)

 <li>The dataset was heavily skewed, some examples of which include:</li>
 <ol>
  <li>Weather condition - VMC:IMC :: 13:1</li>
  <li>Injury Severity - Non-Fatal:Fatal:Incident :: 24:5:1</li>
  <li>Aircraft Damage - Substantial:Destroyed:Minor :: 25:4:1</li><br/>

![Weather Condition Percentages](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/skewed.png)

 </ol>
 <li>Most accidents took place during landing.</li><br/>

![Phase of Flight](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/PhaseOfFlight.png)

 <li>The injury severity has a strong correlation with the aircraft damage.</li><br/>

![Injury Severity and Aircraft Damage](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/Injury%20Severity%20Aircraft%20Damage.png)

 <li>A negative correlation between the weather conditions and aircraft damage was observed.</li><br/>

![Weather Condition and Aircraft Damage](https://raw.githubusercontent.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/main/Supporting%20Images/Weather%20Condition%20and%20Aircraft%20Damage.png)

</ol>

For further insights about the dataset and detailed visualizations, check out the [exploratory analysis notebook](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Notebooks/Data_Extraction_Cleanup.ipynb)

## Predictions

### Predicting Aircraft Damage using DTC

In this section, we attempt to predict the aircraft damage with the help of Scikit-Learn's Decision Tree Classifier model using the following attributes (X):
* Altitude
* Latitude
* Longitude
* Total Uninjured
* Total Passengers

We use these attributes to predict the "Aircraft Damage" (y)

Since the dataset is heavily skewed, it presents the problem of minority classes. Hence, we use imbalanced-learn to oversample the records and balance the classes.

```python

#Near Miss instance for data balancing by oversampling minority classes
nm = NearMiss()
smk = SMOTETomek(random_state = 42)
X_train, y_train = smk.fit_resample(X_train, y_train)
```

<br/>

![Aircraft Damage Balancing](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Supporting%20Images/Balanced%20Aircraft%20Damage.png?raw=true)

After this we use a Decision Tree classifier with a depth of 4 to predict the aircraft damage as a function of the input attributes X.  

![Aircraft Damage Decision Tree](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Supporting%20Images/Aircraft%20Damage%20DTC.png?raw=true)

This results in a classification accuracy of 50% on the test set. To improve te accuracy we optimize the decision tree classifier with the help of Adaptive Boosting.

```python
AdaBoost = AdaBoostClassifier(base_estimator = dectree, 
                              n_estimators = 175, learning_rate = 0.3)
boostModel = AdaBoost.fit(X_train,y_train)

pred = boostModel.predict(X_train)
predictions = metrics.accuracy_score(y_train,pred)
print("prediction accuracy of train set is: ",predictions*100,"%")

pred = boostModel.predict(X_test)
predictions = metrics.accuracy_score(y_test,pred)
print("prediction accuracy of test set is: ",predictions*100,"%")
```

After boosting the decision tree, we achieve a classification accuracy of 68% on the testset.


### Predicting Injury Count using Linear Regression

In this section, we attempt to predict the injury count (fatal, severe, and minor injuries) for a commercial aircraft accident using the following attributes:
* Total Passengers
* Altitude of the Aircraft

Since the altitude was derived from land elevation and phase of flight, we hypothesized that these features would be captured with the use of altitude as a feature alone. Our subsequent trials confirmed this hypothesis.

We then used linear regression to make predictions. However, the results were not encouraging with regard to the goodness of fit, variance, and MSE.  

![Predicting Fatal Injury Count](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Supporting%20Images/Fatal%20Injury%20Prediction.png?raw=true)

### Predicting Injury Severity using DTC

Here we attempt to predict the injury severity for passengers onboard the aircraft with the help of the following features:
* Altitude
* Latitude
* Longitude
* Weather Condition
* Total Passengers

As you had seen earlier, the injury severity is heavily skewed with class imbalances. Hence, we use imbalanced-learn to balance the classes before building the model.

```python
nm = NearMiss()
smk = SMOTETomek(random_state = 42)
a_train, b_train = smk.fit_resample(a_train, b_train)
```

![Fatality Balancing](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Supporting%20Images/Fatality%20Balancing.png?raw=true)

We then use a build the decision tree classifier with a depth of 4 to make predictions with an accuracy of 75% on the testset.  

![Fatality Prediction DTC](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Supporting%20Images/Injury%20Severity%20DTC.png?raw=true)

#### For more details on the feature selection, models, and visualizations, please check out our [notebook](https://github.com/aneezJaheez/Predicting-Aviation-Accidents-and-Injuries/blob/main/Notebooks/Prediction_Models.ipynb).

## The Team
* Aneez Ahmed Jaheezuddin
* [You Zhi Min](https://github.com/zzzhimin)
* [Yeong Wei Xian](https://github.com/wxiannnn)
* [Kenneth Lee](https://github.com/klee046)
