import numpy as np

def getW2VfeatureVecs(questions, model, num_features):
    counter = 0
    error_vecs=[]
    questionFeatureVecs = np.zeros((len(questions),num_features),dtype="float32")
    
    for question in questions:
        
        doc = [word for word in question if word in model.vocab]
        if len(doc)==0:
            k=np.zeros(num_features)
            error_vecs.append(counter)
        else:

            k=np.mean(model[doc], axis=0)
            
        questionFeatureVecs[counter] = k
        counter = counter+1
        
    print("Total scanned rows: {0}".format(counter))
    print("total vectors Initialised to Zero: {0} ".format(len(error_vecs)))
    return questionFeatureVecs