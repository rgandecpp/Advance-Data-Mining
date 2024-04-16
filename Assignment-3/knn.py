#-------------------------------------------------------------------------
# AUTHOR: Ruchitha Gande
# FILENAME: Knn.py
# SPECIFICATION: KNN to classify temp into 11 Classes
# FOR: CS 5990- Assignment #3
# TIME SPENT: 35mins
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
#11 classes after discretization
temperature_intervals = [i for i in range(-22, 40, 6)]

def discretize_temperature(instance):
    """
  Updates the "Temperature (C)" key in a dictionary (instance)
  based on predefined temperature intervals.

  Args:
      instance (dict): A dictionary containing data, including a "Temperature (C)" key.

  Returns:
      dict: The updated dictionary with the discretized temperature value.
  """
    try:
        temperature = instance.get("Temperature (C)")
        if temperature is None:
            raise ValueError("Missing 'Temperature (C)' key in the input data")

        for interval_upper_bound in temperature_intervals:
            if temperature <= interval_upper_bound:
                instance["Temperature (C)"] = interval_upper_bound
                return instance  # Early return after finding the interval

        # If temperature doesn't fall within any interval (out of range), then we discretize to the upper limit (38)
        instance["Temperature (C)"] = interval_upper_bound
        return instance
    except (ValueError) as e:
        print("Error: ", e)
    return instance

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
training_data_path = r"Assignment-3\weather_training.csv"
training_df = pd.read_csv(training_data_path, sep=',', header=0)


discreted_training_df = training_df.apply(discretize_temperature, axis = 1)

# The columns that we will be making predictions with.
y_training = np.array(discreted_training_df["Temperature (C)"]).astype(dtype='int')
X_training = np.array(discreted_training_df.drop(["Temperature (C)","Formatted Date"], axis=1).values)

#reading the test data
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
testing_data_path = r"Assignment-3\weather_test.csv"
test_df = pd.read_csv(testing_data_path, sep=',', header=0)

#update the test class values according to the discretization (11 values only)
discreted_testing_df = test_df.apply(discretize_temperature, axis=1)
y_test = discreted_testing_df["Temperature (C)"].astype(dtype='int')
X_test = discreted_testing_df.drop(["Temperature (C)","Formatted Date"], axis=1).values

#loop over the hyperparameter values (k, p, and w) ok KNN
highest_accuracy = 0
for k in k_values:
    for p in p_values:
        for w in w_values:
            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            correct_predictions = 0
            for test_sample, true_value in zip(X_test, y_test):
                predicted_value = clf.predict(np.array([test_sample]))[0]

                #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
                #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
                percentage_difference = abs(predicted_value - true_value) / abs(true_value) * 100
                if -15 <= percentage_difference <= 15:
                    correct_predictions += 1

            accuracy = correct_predictions / len(y_test)
                        
            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            if highest_accuracy < accuracy:
                highest_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {accuracy}, Parameters: k={k}, p={p}, w= '{w}'")





