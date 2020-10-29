"""
Author: Swapnil Acharya
Date: 10/25/2020
Description: An Implementation of Gaussian Naive Bayes.
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class GaussianNaiveBayes():
    """ This class is an implementation of Naive Bayes Algorithm for Gaussain Distribution.
    
    Attributes:
        __features_mean (dictionary): A private class member that contains feature wise mean given label 
        __features_sdevs (dictionary): A private class member that contains feature wise standard deviation given label
        __unique_labels (list): A private classs member that contains the categories (labels) of the train data
        __trainset_df (dataframe): A private class meber that contains the trainning data set
        __prior_probabilities (dictionary): A private class member that contains prior probabilibies for labels in __unique_labels
    """
    def __init__(self):
        """ This method is the default constructor for this class. THis method intializes private class members
        """
        self.__features_mean = {}
        self.__features_sdevs = {}
        self.__unique_labels = []
        self.__trainset_df = None
        self.__prior_probabilities = {}
    
    def fit(self,train_data_df):
        """ THis function computes the parameters neeeded to calculate gaussian naive bayes probabilities.
        
        Args:
            train_data_df(pd.DataFrame): the trainning data for Gaussian Naive Bayes
            
        Returns:
            None
        """
        #set train set for model 
        self.__trainset_df = train_data_df.copy()
        
        #  get the the column that contains the category 
        label_column = list(self.__trainset_df.columns)
        label_column = label_column[len(label_column)-1]
        labels = train_data_df[label_column].to_numpy() #get labels for prior cacluations
        total_count = len(labels) #compute totalnumber of data points
        
        # get list of class labels
        self.__unique_labels = set(self.__trainset_df[label_column])
        
        #caclulate gaussian parameters
        for label in self.__unique_labels:
            
            # calcuate means and standard deviations
            self.__features_mean[label] = list(self.__trainset_df[self.__trainset_df[label_column]==label].mean()) #get mean for specified classlabel  
            self.__features_sdevs[label] = list(self.__trainset_df[self.__trainset_df[label_column]==label].std()) #get standard deviation for specified classlabel    
    
            #mean and standard deviation for label coumn is not required so drop then
            self.__features_mean[label].pop()
            self.__features_sdevs[label].pop()
            
            #prior probabity calculations
            label_count = len(labels[labels==label])
            prior = (label_count * 1.0) / (total_count * 1.0)
            self.__prior_probabilities[label] = prior


    def print_model_parameters(self):
        """ This class method prints the Gaussain Naive Bayes Model Parameters.
        
        Args:
            None
            
        Returns:
            None
        """
        print("Classes: ",self.__unique_labels) #print categories
        print("")
        print("Means : ", self.__features_mean) #print feature wise mean
        print("")
        print("Standard Deviations: ",self.__features_sdevs) #print feature wise standard deviation
        print("")
        print("Prior Probabilities: ", self.__prior_probabilities) #print label wise prior probabilities
        
    
    def predict_probabilities(self,test_set_df):
        """This class method predits the probabiilities of given test data bsed of the model parameters calculate from trainning data.
        
        Args:
            test_set_df(pandas.DataFrame): A Dataframe that contains the testing set whose probability is to be predicted
            
        Returns:
            predicted_probabilities(pandas.DataFrame): A Dataframe that contains the label wise probability for test set
        """
        
        #get the columnt that contains the label
        column_names = test_set_df.columns
        label_column = len(column_names) - 1
        feature_matrix = test_set_df.drop([label_column],axis=1)
        feature_matrix = feature_matrix.to_numpy()
        
        #intialize lists to store labels and predicited probabilities
        labels_list = []
        probabilities = []
        
        # calculate gaussain likelihoods
        for label in self.__unique_labels:
            cur_matrix = feature_matrix - np.array(self.__features_mean[label])
            cur_matrix = cur_matrix **2
            cur_matrix = cur_matrix / (2 * (np.array(self.__features_sdevs[label])**2) )
            cur_matrix = np.exp((-1 * cur_matrix))
            cur_matrix = cur_matrix * (1.0/np.sqrt(2*np.pi*(np.array(self.__features_sdevs[label]))))
            
            if cur_matrix.ndim > 1:
                cur_matrix = np.prod(cur_matrix,axis=1)
            else:
                cur_matrix = np.prod(cur_matrix)
            
            #multiply by prior probabilities
            cur_matrix = cur_matrix * self.__prior_probabilities[label]
            
            labels_list.append(label)
            probabilities.append(cur_matrix)
        
        # make probabilities into single dataframe
        probabilities_df = pd.DataFrame(probabilities) #convert to dataframe
        probabilities_df = probabilities_df.transpose()
        probabilities_df.columns = labels_list #add labels
        return probabilities_df
    
    def predict_labels(self,test_set_df):
        """ This class method predicts the category label for given test set
        
        Args:
            test_set_df(pandas.DataFrame): A Dataframe that contains the testing set whose label is to be predicted
            
        Returns:
            predicted_labels(numpy.array): A numpy array that contains the predicted labels for given test data
        """
        probabilities_df = self.predict_probabilities(test_set_df) #get predicted probabilities
        labels_list = probabilities_df.columns #add column names 
        probabilities_df = probabilities_df.to_numpy() #convrt dataframe to numpy
        max_index = np.argmax(probabilities_df,axis=1) #get maximum of each row, axis= 1 horizontal, max each row
        predicted_labels = np.array(labels_list[max_index]) #get the value at the max index
        return predicted_labels #return predicted label
        
    def evaluate(self,test_set_df):
        """ This class method evalautes the model performace for given test set, the evaluation metric is accuracy
        
        Args:
            test_set_df(pandas.DataFrame): A Dataframe that contains the testing set whose label is to be predicted
            
        Returns:
            accuracy(float): THe percentage accuracy of model's predictions
        """        
        column_names = test_set_df.columns #set column names
        label_column = len(column_names) - 1
        actual_labels = test_set_df[label_column].to_numpy()#convert from dataframe to numpy
        predicted_labels = self.predict_labels(test_set_df) #get preditions
        #print("1",len(predicted_labels[predicted_labels==1]))
        #print("-1",len(predicted_labels[predicted_labels==-1]))
        agreement =  ( (1.0 * sum(actual_labels == predicted_labels)) / (1.0 * len(predicted_labels)) ) * 100 #compute accuracy
        print("Model Accuracy: %0.3f%%"%agreement) 
        return agreement


def get_dataframe(filename="irisTraining.txt",header=None,delimiter=" "):
    """This function reads the data from given files and return a pandas datafram object.
    
    Args:
        filename(string): The name of file that contains dataset
        header: Header of the file
        delimiter(string): the character that sperates columns of data
    Returns:
        pandas dataframe: A pandas datafram object containing the data from the filename
    """
    dataframe_df = pd.read_csv(filename,sep=delimiter,header=header) #read files
    return dataframe_df #return dataframe


def binary_confusion_matrix(actual_labels,predicted_labels):
    """This function compute, plots and returns a binary confusion matrix given actaul and predicted labels
    
    Args:
        actual_labels(numpy.array): A numpy array that contains actual labels
        predicted_labels(numpy.array): A numpy array that contains predicted labels
        
    Returns:
        confusion matrix(numpy.array): A binary confusion  matrix 
    """
    
    #initialize variables
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0
    #compute tp,fn,fp,tn
    for i in range(0,len(actual_labels)):
        if actual_labels[i]==1:
            if actual_labels[i] == predicted_labels[i]: #1 = 1
                true_positives += 1
            else: #1 != -1
                false_negatives += 1
        else: 
            if actual_labels[i] == predicted_labels[i]: #0 = 0
                true_negatives += 1
            else: #1 != -1
                false_positives += 1
    #arrange tp, fn, fp, tn into an 2x2 array
    confusion_matrix = np.squeeze(np.array([[true_positives,false_negatives],[false_positives,true_negatives]]))
    print("Confusion Matrix")
    for i in range(0,confusion_matrix.shape[0]):
        for j in range(0,confusion_matrix.shape[1]):
            if ((i==0) and (j==0)):
                print("Tp: ",confusion_matrix[i,j]," ",end="")
            elif ( (i==0) and (j==1)):
                print("Fn: ",confusion_matrix[i,j]," ",end="\n")
            elif((i==1) and (j==0)):
                print("Fp: ",confusion_matrix[i,j]," ",end="")
            else:
                print("Tn: ",confusion_matrix[i,j]," ",end="\n")
                
    #plot confusion matrix
    plt.clf()
    #plt.imshow(cm, interpolation='nearest', cmap=plt.cm.inferno)
    plt.imshow(confusion_matrix, cmap=plt.cm.inferno)
    classNames = ['Positive','Negative']
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TP','FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(s[i][j])+"="+str(confusion_matrix[i][j]),color="r",fontsize=12)
    plt.show()
    return confusion_matrix



def binary_aprf(actual_labels,predicted_labels):
    """This function computes and displays the accuray, precision, recall and f-measure given actual and predicted labels
    
    Args:
        actual_labels(numpy.array): A numpy array that contains actual labels
        predicted_labels(numpy.array): A numpy array that contains predicted labels
        
    Returns:
        None
    """
    #initialize variables
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0
    
    #compute tp,fn,fp,tn
    for i in range(0,len(actual_labels)):
        if actual_labels[i]==1:
            if actual_labels[i] == predicted_labels[i]: #1 = 1
                true_positives += 1
            else: #1 != -1
                false_negatives += 1
        else: 
            if actual_labels[i] == predicted_labels[i]: #0 = 0
                true_negatives += 1
            else: #1 != -1
                false_positives += 1
    #compute metrics            
    accuracy = ((true_positives + true_negatives) *1.0) / (1.0 * (true_positives + true_negatives + false_negatives + false_positives) )
    precision = (true_positives * 1.0) / ( (true_positives + false_positives)* 1.0)
    recall = (true_positives * 1.0) / ( (true_positives + false_negatives)* 1.0)
    f_measure = (2.0 * true_positives) / (1.0 * ( (2.0 * true_positives) + false_positives + false_negatives ) )
    
    #display metrics
    print("Accuracy: %0.2f"%accuracy)
    print("Precision: %0.2f"%precision)
    print("Recall: %0.2f"%recall)
    print("F-Measure: %0.2f"%f_measure)


def main():
    trainingFile = "datasets/irisPCTraining.txt"
    testingFile = "datasets/irisPCTesting.txt" 
    train_df = get_dataframe(filename=trainingFile,header=None,delimiter=" ")
    test_df = get_dataframe(filename=testingFile,header=None,delimiter=" ")
    print("Training Data: " + trainingFile + "\n" + "Testing Data: " + testingFile)
    print("")
    model = GaussianNaiveBayes()
    model.fit(train_df)
    print("Fitted Model Prameters: \n")
    model.print_model_parameters()
    print("")
    print("Predicted Probabilities On Test Data: \n")
    print(model.predict_probabilities(test_df))
    print("")
    predicted_labels = model.predict_labels(test_df)
    columns_list  = list(test_df.columns)
    label_column = len(columns_list) - 1
    data_df = pd.DataFrame()
    data_df["Actual Labels"] = test_df[label_column].to_numpy()
    data_df["Predicted Labels"] = predicted_labels
    print(data_df)
    print("")
    print("Confusion Matrix of Test Data")
    binary_confusion_matrix(test_df[label_column],predicted_labels)
    print("")
    print("Metrics of Test Data:")
    binary_aprf(test_df[label_column],predicted_labels)
    

if __name__ == "__main__":
    main()


