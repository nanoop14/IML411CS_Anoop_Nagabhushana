from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import *
from bs4 import BeautifulSoup 
import re # For regular expressions
from nltk.corpus import stopwords
import nltk.data
import nltk
from pylab import *
import numpy as np
import pandas as pd


def getSplits(x,y):

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)
    print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),(len(x_train[y_train == 0]) / (len(x_train)*1.))*100,(len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
    print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation), (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100, (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
    print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),(len(x_test[y_test == 0]) / (len(x_test)*1.))*100, (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))
    return x_train.values,y_train.values,x_validation.values,y_validation.values,x_test.values,y_test.values

def clean_questions(question, remove_stopwords=False):
    # 1. Removing html tags
    question_text = BeautifulSoup(question).get_text()
    # 2. Removing non-letter.
    question_text = re.sub("[^a-zA-Z]"," ",question_text)
    # 3. Converting to lower case and splitting
    words = question_text.lower().split()
    # 4. Optionally remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))     
        words = [w for w in words if not w in stops]
    
    return(words)

def cleanData(questions,remove_stopwords=False):
    clean_data = []
    for question in questions:
        clean_data.append(clean_questions(question, remove_stopwords))
    return clean_data


def getModelAccuracy(y_true,ypred,labels,print_details=False):
    confmat = np.array(confusion_matrix(y_true, ypred, labels=labels))
    confusion = pd.DataFrame(confmat, index=['positive', 'negative'],
                             columns=['predicted_positive','predicted_negative'])
    if print_details:
        print ("Accuracy Score: {0:.2f}%".format(accuracy_score(y_true, ypred)*100))
        print ("-"*80)
        print ("Confusion Matrix\n")
        print (confusion)
        print ("-"*80)
        print ("Classification Report\n")
        print (classification_report(y_true, ypred))
    return accuracy_score(y_true, ypred)*100

def getBaselineAccuracy(x_train,y_train,x_validation,y_validation):
    neg=(len(x_train[y_train == 0]) / (len(x_train)*1.))*100
    pos=(len(x_train[y_train == 1]) / (len(x_train)*1.))*100
    N=x_validation.shape
    if(neg>pos):
        ypred=np.zeros(N)
        return neg
    else:
        ypred=np.ones(N)
        return pos

def getPlot(X,Y,x_label,y_label,plt_label,colour='b',location=''):
    rainPlt= plt.plot(X,Y,'-',c=colour,label=plt_label)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    if len(location)>0:
        plt.savefig(location)
    return rainPlt
