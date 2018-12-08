import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import util
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def run_nn(x_train,y_train,x_val,y_val):
    seed=7
    np.random.seed(seed)
    #model for nn:
    model_nn_with_stop_words = tf.keras.Sequential()
    model_nn_with_stop_words.add(layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu', input_dim=300))
    model_nn_with_stop_words.add(layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
    model_nn_with_stop_words.add(layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
    model_nn_with_stop_words.add(layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
    model_nn_with_stop_words.add(layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
    model_nn_with_stop_words.add(layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
    model_nn_with_stop_words.add(layers.Dense(1, activation='sigmoid'))
    model_nn_with_stop_words.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history=model_nn_with_stop_words.fit(x_train,y_train,
                     validation_data=(x_val, y_val),
                     epochs=15, batch_size=32, verbose=2)
    print(history.history.keys())
    print("saving nn model:~/model_performance/Neural_network/nn")
    model_nn_with_stop_words.save(filepath='model_performance/Neural_network/nn',overwrite=True)
    history.history.keys()
    plt.plot(model_nn_with_stop_words.history.history['acc'])
    plt.plot(model_nn_with_stop_words.history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.show()
    return history

def run_lr(x_train,y_train,x_val,y_val):
    #use this to tune params:
    possible_c_values=[1,2]
    lr_accuracy=[]
    train_accuracy=[]

    for c in possible_c_values:
        lr=LogisticRegression(solver='sag',random_state=200,C=c)
        lr.fit(x_train,y_train)
        lr_pred=lr.predict(x_val)
        print("For c: {0}".format(c))
        lr_accuracy.append(util.getModelAccuracy(y_val,lr_pred,np.array([0,1]),print_details=True))
        train_accuracy.append(util.getModelAccuracy(y_train,lr.predict(x_train),np.array([0,1])))
    util.getPlot(possible_c_values,lr_accuracy,"Lr c values","accuracy","validation acc",colour='b')
    util.getPlot(possible_c_values,train_accuracy,"Lr c values","accuracy","train acc",colour='r')

def run_adaboost(x_train,y_train,x_val,y_val):
    #use this to tune params:
    possible_c_vlaues=[1,2]
    ada_accuracy=[]
    ada_train_accuracy=[]
    for c in possible_c_vlaues:
        ada=AdaBoostClassifier(n_estimators = c)
        ada.fit(x_train,y_train)
        print("For c: {0}".format(c))
        ada_pred=ada.predict(x_val)
        ada_accuracy.append(util.getModelAccuracy(y_val,ada_pred,np.array([0,1]),print_details=True))
        ada_train_accuracy.append(util.getModelAccuracy(y_train,ada.predict(x_train),np.array([0,1])))

    util.getPlot(possible_c_vlaues,ada_accuracy,"ADABoost number of trees","accuracy","validation acc",colour='b')
    util.getPlot(possible_c_vlaues,ada_train_accuracy,"ADABoost number of trees","train acc",colour='r')

def run_rf(x_train,y_train,x_val,y_val):
    #use this to tune params:
    possible_c_values=[1,2]
    rf_accuracy=[]
    train_accuracy=[]
    for c in possible_c_values:
        rf=RandomForestClassifier(n_estimators=c,max_depth=7)
        rf.fit(x_train,y_train)
        rf_pred=rf.predict(x_val)
        print("For c: {0}".format(c))
        rf_accuracy.append(util.getModelAccuracy(y_val,rf_pred,np.array([0,1]),print_details=True))
        train_accuracy.append(util.getModelAccuracy(y_train,rf.predict(x_train),np.array([0,1])))
    util.getPlot(possible_c_values,rf_accuracy,"Rf c values for depth 7","accuracy","validation acc",colour='b')
    util.getPlot(possible_c_values,train_accuracy,"Rf c values for depth 7","accuracy ","train acc",colour='r')
