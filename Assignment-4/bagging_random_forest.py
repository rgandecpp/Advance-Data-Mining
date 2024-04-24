#-------------------------------------------------------------------------
# AUTHOR: Ruchitha Gande
# FILENAME: bagging_random_forest.py
# SPECIFICATION: ensemble and random forest implementation and accuracy
# FOR: CS 5990- Assignment #4
# TIME SPENT: 45mins
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
training_data_path = r"Assignment-4\optdigits.tra"
dbTraining = pd.read_csv(training_data_path, sep=',', header=None)

#reading the test data from a csv file and populate dbTest
test_data_path = r"Assignment-4\optdigits.tes"
dbTest = pd.read_csv(test_data_path, sep=',', header=None)

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
classVotes=[0,0,0,0,0,0,0,0,0,0]

print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

  #populate the values of X_training and y_training by using the bootstrapSample
  X_training = np.array(bootstrapSample.values)[:,:64]
  y_training = np.array(bootstrapSample.values)[:,-1]

  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)

  correct_predictions = 0
  for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      model_predicted = clf.predict([dbTest.iloc[i,:64]])
      model_predicted = int(model_predicted[0])
      classVotes[model_predicted] += 1
      original = dbTest.iloc[i,-1]

      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         if model_predicted == original:
            correct_predictions += 1

  if k == 0: #for only the first base classifier, print its accuracy here
     accuracy = correct_predictions / len(dbTest.index)
     print("Finished my base classifier (fast but relatively low accuracy) ...")
     print("My base classifier accuracy: " + str(accuracy))
     print("")

#now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)

# Calculate ground truth label counts
ground_truth_label_counts = [0] * 10
for label in dbTest.iloc[:, -1]:
    ground_truth_label_counts[int(label)] += 1

# Calculate error count
errors = sum(abs(classVotes[i] - ground_truth_label_counts[i]) for i in range(10))

# Calculate accuracy
accuracy = 1 - (errors / len(dbTest))

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
test_attributes = np.array(dbTest.values)[:,:64]    #getting the first 64 fields to form the feature data for test
test_class_label = np.array(dbTest.values)[:,-1]
correct_predictions = 0
for attributes, class_label in zip(test_attributes, test_class_label):
   model_predicted = clf.predict([attributes])[0]


#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
   if model_predicted == class_label:
      correct_predictions+=1

#printing Random Forest accuracy here
accuracy = correct_predictions/len(dbTest.index)
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
