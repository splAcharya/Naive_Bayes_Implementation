"""
Author: Swapnil Acharya
Date: 10/25/2020
Description: An Implementation of Categorical Naive Bayes.
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



class CategoricalNaiveBayes():
    """ This class is an implementation of Naive Bayes Algorithm for Categorial Data.
    
    Attributes:
        __unique_labels (list): A private classs member that contains the categories (labels) of the train data
        __label_frequencies(dictionary): A private class member that contains the frequencies of category labels,
        __category_frequencies(dictionary): A private class member that contains the features of feature-value given class
        __prior_probabilties(dictionary): A private class member that contains the prior probabilties for all category labels
        __posterior_probabilities(dictionary): A private class memebr that contains the feature-value given label probabilitiy for all feature-values
        __featurewise_unique_count(dictionary): A private class member that contains count of unique value per feature
    """
    
    def __init__(self):
        """This class method is the default constructor. This initializes all private class memebers
        """
        self.__unique_labels = ()
        self.__label_frequencies = {}
        self.__category_frequencies = {}
        self.__prior_probabilities = {}
        self.__posterior_probabilities = {}
        self.__featurewise_unique_count = {}
        self.__laplacian_smoothing_constant = 1
        
    
    def print_model_parameters(self):
        """This class method prints the trained model prameters for Categorial Naive Bayes
        """
        print("Classes: ", self.__unique_labels) #print unique labels
        print("")
        print("Feature Wise Unique Count: ", self.__featurewise_unique_count) #print feature wise unique count
        print("")
        print("Class Frequencies: ",self.__label_frequencies) #print label frequencies
        print("")
        print("Prior Probabilities: ",self.__prior_probabilities) #print prior probabilties
        print("")
        print("Category Frequencies: ") #print categorical and posterior proabilities frequencies
        for key_pair in self.__category_frequencies.keys():
            print(key_pair," : ",self.__category_frequencies[key_pair])
        print("")
        print("Laplacian Smoothing Constant: ", self.__laplacian_smoothing_constant)
        print("")
        print("Posterior Probabilties With Laplacian Smoothing: ") #print posterior probabilties
        for key_pair in self.__posterior_probabilities.keys():
            print(key_pair," : ",self.__posterior_probabilities[key_pair])


    def fit(self,train_set_df):
        """ THis function computes the parameters neeeded to calculate categorical naive bayes probabilties.
        
        Args:
            train_data_df(pd.DataFrame): the trainning data for Gaussian Naive Bayes
            
        Returns:
            None
        """
        #convert from datafram to numpy
        feature_matrix= train_set_df.to_numpy()
        
        #compute label frequencies and prior probabilities
        label_column = list(train_set_df.columns)
        label_column = label_column[len(label_column)-1]
        self.__unique_labels = list(set(train_set_df[label_column])) # get list of class labels
        
        #compute label frequencies and prior probabilities
        total_labels = len(train_set_df[label_column]) #total count of labels 
        for label in self.__unique_labels:
            freq = len(train_set_df[train_set_df[label_column]==label][label_column])
            self.__label_frequencies[label] = freq #compute label frequency
            self.__prior_probabilities[label] = (1.0 * freq) / (total_labels * 1.0) #compute prior probabiity
        
        #create a hastable that has counts of feature-value given class
        for i in range(0,feature_matrix.shape[0]):
            for j in range(0,feature_matrix.shape[1]-1):
                attribute = j 
                category = feature_matrix[i,j]
                label = feature_matrix[i,feature_matrix.shape[1]-1]
                key_pair = (attribute,category,label)
                if key_pair not in self.__category_frequencies:
                    self.__category_frequencies[key_pair] = 1
                else:
                    self.__category_frequencies[key_pair] += 1
                    
        #count number of unique values per feature, as required by laplacian smoothing
        for col in range(0,feature_matrix.shape[1]-1):
            self.__featurewise_unique_count[col] = len(set(feature_matrix[:,col]))
        
        #compute posterior likelihood and include laplacian smoothing
        for key_pair in self.__category_frequencies.keys():
            for label in self.__unique_labels:
                if label == key_pair[2]: #if categorical frequencies have already been calculate then 
                    posterior_prob_numerator = self.__category_frequencies[key_pair] + self.__laplacian_smoothing_constant #numerator of posterior probability
                    posterior_prob_denominator = self.__label_frequencies[key_pair[2]]+ self.__featurewise_unique_count[key_pair[0]] #denominator of posterior probability
                    posterior_prob = (1.0 * posterior_prob_numerator)  / (1.0 * posterior_prob_denominator) #compute posterior probabilitiy
                    self.__posterior_probabilities[key_pair] = posterior_prob #save the posteriro probability
                    
                elif (key_pair[0],key_pair[1],label) not in self.__posterior_probabilities.keys(): #else create new key to prevent zero probabilties
                    posterior_prob_numerator = 0 + self.__laplacian_smoothing_constant #numerator of posterior probability
                    posterior_prob_denominator = self.__label_frequencies[label]+ self.__featurewise_unique_count[key_pair[0]] #denominator of posterior probability
                    posterior_prob = (1.0 * posterior_prob_numerator)  / (1.0 * posterior_prob_denominator) #compute posterior probabilitiy
                    self.__posterior_probabilities[(key_pair[0],key_pair[1],label)] = posterior_prob #save the posteriro probability

                    
    def predict_probabilities(self,test_set_df):
        """This class method predits the probabiilities of given test data bsed of the model parameters calculate from trainning data.
        
        Args:
            test_set_df(pandas.DataFrame): A Dataframe that contains the testing set whose probability is to be predicted
            
        Returns:
            predicted_probabilities(pandas.DataFrame): A Dataframe that contains the label wise probability for test set
        """
        
        #get column
        label_column = list(test_set_df.columns)
        label_column = label_column[len(label_column)-1]
        feature_matrix = test_set_df.to_numpy() #convert to numpy
        predicted_probailities = [] #predcited probabilities
        
        #compute probablities for all test set
        for i in range(0,feature_matrix.shape[0]):
            row_probs = np.zeros(len(self.__unique_labels))
            for j in range(0,feature_matrix.shape[1]-1):
                attribute = j
                category = feature_matrix[i,j]
                label_wise_prob = []
                for k in range(0,len(self.__unique_labels)):
                    key_pair = (attribute,category,self.__unique_labels[k])
                    posterior_prob = self.__posterior_probabilities[key_pair]
                    if j == 0:
                        row_probs[k] = posterior_prob * self.__prior_probabilities[self.__unique_labels[k]]
                    else:
                        row_probs[k] *= posterior_prob
            predicted_probailities.append(row_probs)
            
        predicted_probailities = np.squeeze(np.array(predicted_probailities))
        predicted_probailities = pd.DataFrame(predicted_probailities) #convert to dataframe
        predicted_probailities.columns = self.__unique_labels
        return predicted_probailities
        
        
    def predict_labels(self,test_set_df):
        """ This class method predicts the category label for given test set
        
        Args:
            test_set_df(pandas.DataFrame): A Dataframe that contains the testing set whose label is to be predicted
            
        Returns:
            predicted_labels(numpy.array): A numpy array that contains the predicted labels for given test data
        """
        probabilities_df = self.predict_probabilities(test_set_df)
        labels_list = probabilities_df.columns
        probabilities_df = probabilities_df.to_numpy()
        max_index = np.argmax(probabilities_df,axis=1) #get maximum of each row, axis= 1 horizontal, max each row
        predicted_labels = np.array(labels_list[max_index])
        return predicted_labels
    
    def evaluate(self,test_set_df):
        """ This class method evalautes the model performace for given test set, the evaluation metric is accuracy
        
        Args:
            test_set_df(pandas.DataFrame): A Dataframe that contains the testing set whose label is to be predicted
            
        Returns:
            accuracy(float): THe percentage accuracy of model's predictions
        """    
        column_names = test_set_df.columns
        label_column = len(column_names) - 1
        actual_labels = test_set_df[label_column].to_numpy()
        predicted_labels = self.predict_labels(test_set_df)
        #print("1",len(predicted_labels[predicted_labels==1]))
        #print("-1",len(predicted_labels[predicted_labels==-1]))
        agreement =  ( (1.0 * sum(actual_labels == predicted_labels)) / (1.0 * len(predicted_labels)) ) * 100
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
    trainingFile = "datasets/buyTraining.txt"
    testingFile = "datasets/buyTesting.txt" 
    train_df = get_dataframe(filename=trainingFile,header=None,delimiter=" ")
    test_df = get_dataframe(filename=testingFile,header=None,delimiter=" ")
    print("Training Data: " + trainingFile + "\n" + "Testing Data: " + testingFile)
    print("")
    model = CategoricalNaiveBayes()
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
 
    