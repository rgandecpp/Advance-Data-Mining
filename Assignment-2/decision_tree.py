# -------------------------------------------------------------------------
# AUTHOR: Ruchitha Gande
# FILENAME: Decision_tree
# SPECIFICATION: decision tree to predict cheat or not
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1hr 10mins
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

def transform_record_attributes(row):
    marital_encoding = {"Single": [1, 0, 0], "Married": [0, 1, 0], "Divorced": [0, 0, 1]}
    new_row = []
    for ind, val in enumerate(row):
        if ind == 0:  
            # its Refund, so Yes -> 1 and No -> 0
            if val == "Yes":
                new_row.append(1)
            else:
                new_row.append(0)
        elif ind == 1:  
            # Marital Status, one hot encoding
            new_row.extend(marital_encoding[val])
        elif ind == 2:  
            # Taxable Income, converting to float
            new_row.append(float(val.replace("k", "")))
    return new_row

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv("Assignment-2/"+ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    X = []
    for row in data_training:
        new_row = transform_record_attributes(row)
        X.append(new_row) 

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y = []
    for each in data_training:
        if each[-1] == "Yes":
            Y.append(1)
        else:
            Y.append(2)
    #loop your training and test tasks 10 times here
    accuracies = []
    for i in range (10):

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        #read the test data and add this data to data_test NumPy
        #--> add your Python code here
        test_df = pd.read_csv('Assignment-2/cheat_test.csv', sep=',', header=0)
        data_test = np.array(test_df.values)[:,1:]

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data in data_test:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            test_row = transform_record_attributes(data)
            predicted_class = clf.predict([test_row])[0]
            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            if data[3] ==" Yes":
                actual_class = 1
            else:
                actual_class = 2

            if predicted_class == actual_class:
                if predicted_class == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted_class == 1:
                    fp += 1
                else:
                    fn += 1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy)
        #find the average accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
    avg_accuracy = sum(accuracies)/len(accuracies)
    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here
    print("final accuracy when training on "+ ds +": ",  avg_accuracy)


