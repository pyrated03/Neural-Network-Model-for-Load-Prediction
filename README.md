# Neural-Network-Model-for-Loan-Prediction

### Data Analysis, Imputing the Data
We are using the KNN Algorithm for data imputation. Each sample’s missing values are imputed using the mean value from ‘n_neighbours’ nearest neighbours (in our model, we have set ‘n_neighbours’  to 10) found in the training set. 

The basic logic is to consider two samples close, if the features that neither is missing are close. We are using KNNImputer(), a library function in the “impute” module of sklearn.

For imputing the feature matrix(train_x), we first measure the Euclidean distance to find the closest neighbours(i.e. The data points similar to the one considered), and the missing values can be estimated by averaging the values of neighboring observations, corresponding to the missing value.

We then use the completed feature matrix to impute and fill any missing labels.

Hence, we finally get a completed feature matrix and label vector, which can be used to train the model.



### Model Definition and Training 
We are using Keras from Tensorflow for defining and training the model (Sequential Neural Network).

We have divided the total dataset, 75% for training and 25% for testing the model. We used train_test_split(), a library function in sklearn, for this purpose. We are using 10% of the training data(75% of the total dataset) for cross validation.

We tried many types of neural networks, by changing the number of hidden layers, and the number of neurons in each layer to find the best combination, based on the cross validation accuracy. Through repeated checking, we have concluded that a 3-layer neural network, with the first layer having 50 neurons, 2nd having 20, and the 3rd having 8 neurons, all layers densely connected, works the best for the application.


Since the dataset given is highly imbalanced (around 94% of the data is of label ‘Not Default’, i.e. label=0), a class weight is introduced. This will give a higher penalty for wrong classification of the minority class

The model is trained using the training data.

Since we finally want a binary classification model, the loss has been defined to be binary_crossentropy.

### Hyper-Parameter Tuning and Validation 

We have tuned all the hyper parameters, by considering the cross validation and test data accuracy(the 25% of the data that was taken from the total dataset for test). Using this concept, we have concluded that the following combination gives the best accuracy(both cross validation and test):
kernel_initializer = 'uniform’
optimizer="adam"
batch_size=10
validation_split=0.1 
10% of the training data
(75% of the total dataset) for cross validation
activity_regularizer=l1 (0.005)
L1 regularization is similar to Lasso Regularization.
(0.005 is the regularization parameter)
 
The activation function for all 3 layers is ‘ReLu’
The activation function for the output layer has to be ‘sigmoid’ since it is a  binary classifier.
A Dropout layer with dropout rate = 0.1 has been used.
Early Stopping was also used so that the training is time efficient.

**Finally, after the fine tuning, the model reached the highest accuracy of 98.59%.**

