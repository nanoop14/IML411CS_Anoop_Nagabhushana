import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import data
from sklearn.externals import joblib
import util

def test_models(x,y):    
    rf=joblib.load('model_performance/Random_forest/rf_model.sav')
    lr=joblib.load('model_performance/Logistic_Regression/lr_model.sav')
    adaBoost=joblib.load('model_performance/AdaBoost/adaBoost_model.sav')
    nn_mod=tf.keras.models.load_model('model_performance/Neural_network/nn')
    
    print("testing for Baseline Model:")
    util.getModelAccuracy(labels=np.array([0,1]),ypred=np.zeros(x.shape[0]),y_true=y,print_details=True)
    print("testing for Neural network Model:")
    util.getModelAccuracy(labels=np.array([0,1]),ypred=nn_mod.predict_classes(x),y_true=y,print_details=True)
    print("testing for Random Forest Model:")
    util.getModelAccuracy(labels=np.array([0,1]),ypred=rf.predict(x),y_true=y,print_details=True)
    print("testing for Logistic regression Model:")
    util.getModelAccuracy(labels=np.array([0,1]),ypred=lr.predict(x),y_true=y,print_details=True)
    print("testing for AdaBoost Model:")
    util.getModelAccuracy(labels=np.array([0,1]),ypred=adaBoost.predict(x),y_true=y,print_details=True)

def demo_run_all_models(x,y):
    if x is None:
        x,y=data.load_preprocessed_test_data()
    test_models(x,y)
    