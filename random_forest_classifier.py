'''
Random Forest Classifier for Bee Wing Identification

Reads in csv file with list of features as well as labeled class,
and uses a Random Forest Classifer to train and test the data.

'''

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


train_ratio = 0.6

#Loads in data, separates out species labels
bee_data = pd.read_csv('bee_info.csv')
labels = bee_data['species']

# #Splits up the bee_data into training and test set, factorizes it into numbers, instead of species
bee_data['is_train'] = np.random.uniform(0, 1, len(bee_data)) <= train_ratio
train_data, test_data = bee_data[bee_data['is_train']==True], bee_data[bee_data['is_train']==False]
features = bee_data.columns[:3]
factorized = pd.factorize(train_data['species'])
training_labels, testing_labels, classes = factorized[0], pd.factorize(test_data['species'])[0], factorized[1]
train, test = train_data[features], test_data[features]


# #Creates a RandomForestClassifier, fits it on training data, and predicts on test data
rf = RandomForestClassifier(random_state=42)
rf.fit(train, training_labels)
predictions = rf.predict(test)

#Calculates accuracy of the prediction on the test data
accuracyRF = accuracy_score(testing_labels, predictions)
print("Random Forest Classifier Accuracy: ", accuracyRF)

#Displays a confusion of classified species
print("\nConfusion Matrix: ")
preds = classes[predictions]
testing_labs = classes[testing_labels]
confusion_mat = pd.crosstab(testing_labs, preds, rownames = ['Actual Species'], colnames = ['Predicted Species'])
print(confusion_mat)
