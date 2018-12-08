import numpy as np
import pandas as pd
import gensim
import wordEmbeddingUtil
import pprint

#loads only the testing data which is already Preprocessed.
def load_preprocessed_test_data():
    x_test_feature_vec_with_stop_words=np.load("inp/test/w2v_vectors/x_test_feature_vec_with_stop_words.npy")
    y_test_with_stop_words=np.load("inp/test/y_test_with_stop_words.npy")

    return x_test_feature_vec_with_stop_words,y_test_with_stop_words
#Loads the preprocessed data from the directory
def load_preprocessed_data():
    
    x_train_feature_vec_with_stop_words=np.load('inp/train/w2v_vectors/x_train_feature_vec_with_stop_words.npy')
    y_train_with_stop_words=np.load('inp/train/y_train_with_stop_words.npy')

    x_validation_feature_vec_with_stop_words=np.load('inp/validation/w2v_vectors/x_validation_feature_vec_with_stop_words.npy')
    y_validation_with_stop_words=np.load('inp/validation/y_validation_with_stop_words.npy')

    x_test_feature_vec_with_stop_words,y_test_with_stop_words=load_preprocessed_test_data()
    
    return x_train_feature_vec_with_stop_words,y_train_with_stop_words,x_validation_feature_vec_with_stop_words,y_validation_with_stop_words,x_test_feature_vec_with_stop_words,y_test_with_stop_words

#naalyses the data provided to it.
def analyse_data(train):
    pd.set_option('display.width', 1000)
    print("Analysing the given data:\n")
    print("Data consists of: ")
    print(train.columns.values)
    print("\nShape of data:")
    print(train.shape)
    print("\nsample data:")
    pprint.pprint(train.head())
    zero_data=train[train["target"]==0]
    one_data=train[train["target"]==1]
    zero_data_count=zero_data.shape[0]
    one_data_count=one_data.shape[0]
    one_data_percentage=round(one_data_count/(one_data_count+zero_data_count),3)
    zero_data_percentage=1-one_data_percentage
    print("\nClass 1 represents Non Neutral and class 0 represents neutral question")
    print("\npercentage data for class 0: {0}".format(zero_data_percentage*100))
    print("percentage data for class 1: {0}".format(one_data_percentage*100))

#processes the data and spilts into train,dev,test 
def preprocess_data(train):

    #data_analysis(train)
    pd.set_option('display.width', 1000)
    print("Analysing the given data:\n")
    print("Data consists of: ")
    print(train.columns.values)
    print("\nShape of data:")
    print(train.shape)
    print("\nsample data:")
    pprint.pprint(train.head())
    zero_data=train[train["target"]==0]
    one_data=train[train["target"]==1]
    zero_data_count=zero_data.shape[0]
    one_data_count=one_data.shape[0]
    one_data_percentage=round(one_data_count/(one_data_count+zero_data_count),3)
    zero_data_percentage=1-one_data_percentage
    print("\nClass 1 represents Non Neutral and class 0 represents neutral question")
    print("\npercentage data for class 0: {0}".format(zero_data_percentage*100))
    print("percentage data for class 1: {0}".format(one_data_percentage*100))
    
    zero_sampled_data=zero_data.sample(n=one_data_count)
    sampled_train_data=zero_sampled_data.append(one_data,ignore_index=True)
    print("\nSince we have class 1 as minority class with {0} rows, so we have sampled the majority class to obtain a balanced data set of {1}".format(one_data_count,one_data_count*2))
    sampled_train_data.to_csv("sampled_train_data")
    
    #clean data:
    cleaned_data_with_stop_words=util.cleanData(sampled_train_data["question_text"])
    d = {'question_text': cleaned_data_with_stop_words, 'target':sampled_train_data['target'] }
    df_with_stop_words=pd.DataFrame(data=d)
    
    #splitting: 
    print("Dividing data with stop Words into 80-10-10 for train-validation-test\n")
    x_train_with_stop_words,y_train_with_stop_words,x_validation_with_stop_words,y_validation_with_stop_words,x_test_with_stop_words,y_test_with_stop_words=util.getSplits(df_with_stop_words["question_text"],df_with_stop_words["target"])
        
    #save data:
    df_with_stop_words.to_csv("inp/df_with_stop_words")
    
    np.save("inp/train/x_train_with_stop_words",x_train_with_stop_words)
    np.save("inp/train/y_train_with_stop_words",y_train_with_stop_words)
    np.save("inp/validation/x_validation_with_stop_words",x_validation_with_stop_words)
    np.save("inp/validation/y_validation_with_stop_words",y_validation_with_stop_words)
    np.save("inp/test/x_test_with_stop_words",x_test_with_stop_words)
    np.save("inp/test/y_test_with_stop_words",y_test_with_stop_words)
    
    #vectorize data:
    x_train_feature_vec_with_stop_words=wordEmbeddingUtil.getW2VfeatureVecs(x_train_with_stop_words,model,300)
    x_validation_feature_vec_with_stop_words=wordEmbeddingUtil.getW2VfeatureVecs(x_validation_with_stop_words,model,300)
    x_test_feature_vec_with_stop_words=wordEmbeddingUtil.getW2VfeatureVecs(x_test_with_stop_words,model,300)

    #saving vectorized data:
    np.save("inp/train/w2v_vectors/x_train_feature_vec_with_stop_words",x_train_feature_vec_with_stop_words)
    np.save("inp/test/w2v_vectors/x_test_feature_vec_with_stop_words",x_test_feature_vec_with_stop_words)
    np.save("inp/validation/w2v_vectors/x_validation_feature_vec_with_stop_words",x_validation_feature_vec_with_stop_words)

    return x_train_feature_vec_with_stop_words,y_train_with_stop_words,x_validation_feature_vec_with_stop_words,y_validation_with_stop_words,x_test_feature_vec_with_stop_words,y_test_with_stop_words


