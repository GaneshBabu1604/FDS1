# -*- coding: utf-8 -*-
"""
Random Forest Classifier code to classify radio observations of the 
ionosphere into radio observations that observed the ionosphere, or radio 
observations that penetrated into observing interplanetary space (good or bad).

@author: Ganesh Babu
"""

#=====Import Libraries=====#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

#=====Data Reading=====#
# Identify the directory path of this python program
dir_path = os.path.dirname(os.path.realpath(__file__))

# Read in a csv to pandas dataframe
df = pd.read_csv(dir_path + '/ionosphere.data', header = None)

#=====Data Processing=====#
# Recategorize the classification labels into a binary classifier
categ = {"g" : 1, "b" : 0}
df[34] = [categ[item] for item in df[34]]

# Seperate inputs and outputs
labels = df[34]
inputs = df.drop([34],axis=1)

# Seperate data into a training/test set
input_train, input_test, output_train, output_test = train_test_split(inputs, labels, test_size=0.33, random_state=42)

# %%OverSampling the training data to handle class imbalance and performing principal component analysis for dimensionality reduction

print("Before OverSampling, counts of label '1': {}".format(sum(output_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(output_train==0)))

sm = SMOTE(random_state = 27, sampling_strategy = 1.0)
input_train_res, output_train_res = sm.fit_sample(input_train, output_train.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(output_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(output_train_res==0)))

#Dimentioanlity reduction
from sklearn.decomposition import PCA
pca = PCA() 

input_train = pca.fit_transform(input_train)
input_test = pca.transform(input_test)
total=sum(pca.explained_variance_)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Ionosphere Dataset Explained Variance')
plt.show()

k=0
current_variance=0
while current_variance/total < 0.90:
    current_variance += pca.explained_variance_[k]
    k=k+1

#Apply PCA with n_components
pca = PCA(n_components=k)

print(input_train.shape)
input_train = pca.fit_transform(input_train)
input_test = pca.transform(input_test)
print(input_train.shape)

# %% Model Training
#=====Machine Learning Architecture=====#
from sklearn.ensemble import RandomForestClassifier

# Establish a blank slate Gaussian Naive Bayes
model = RandomForestClassifier()

# Train the ML model on the traingin inputs and outputs
hist = model.fit(input_train,output_train)

# Predict, using the ML model, the outputs of the test set
predictions = model.predict(input_test)

# %% Model Validation
#=====Analysis/Validation of ML Model=====#
accuracies = cross_val_score(estimator = model, X = input_train, y = output_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))

# Accuracy
print ("\naccuracy_score :",accuracy_score(output_test, predictions))
    
# Classification report
print ("\nclassification report :\n",(classification_report(output_test, predictions)))


# Create and store a confusion matrix of test data for analysis
SVC_conf = confusion_matrix(output_test,predictions)

print(SVC_conf)

# Plot out the confusion matrix in a nice style
import seaborn as sns
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(confusion_matrix(output_test,predictions),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
ax.set_ylim([0,2])
plt.xlabel("Predicted",fontsize=20)
plt.ylabel("Observed",fontsize=20)
plt.title("CONFUSION MATRIX",fontsize=20)


TP = SVC_conf[1][1]
TN = SVC_conf[0][0]
FN = SVC_conf[1][0]
FP = SVC_conf[0][1]

SVC_HSS = 2.*((TP*TN) - (FN*FP))/((TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)) # Heidke Skill Score
TSS = (TP/(TP + FN)) - (FP/(FP + TN)) # True Skill Score
BIAS = (TP + FP)/(TP + FN) # Bias of Detection
POD = TP/(TP + FN) # Probability of Detection
POFD = FP/(TN + FP) # Probablity of False Detection
FAR = FP/(TP + FP) # False Alarm Ratio
TS = TP/(TP + FN + FP) # Threat Score
OR = (TP*TN)/(FN*FP) # Odds Ratio
print([SVC_HSS,TSS,BIAS,POD,POFD,FAR,TS,OR])
