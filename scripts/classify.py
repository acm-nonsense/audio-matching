import numpy as np
import os
import sys
from sklearn import svm

def process_files(output_root):
    files = os.listdir(output_root)
    values = np.ndarray((len(files),1))
    lables = np.array(["" for _ in range(len(files))],dtype=object)
    for i in range(len(files)):
        values[i] = project_on_average(np.load(output_root+files[i]))
        lables[i] = files[i][:-9]
    return (values,lables)

def project_on_average(sim_mat):
    return np.mean(sim_mat)

def train_model(values,labels):
    model = svm.SVC()
    model.fit(values,labels)
    return model

def predict_labels(values,model):
    return model.predict(values)

values,lables = process_files(sys.argv[1])
print(values)
test = np.array([[0],[0.1],[0.2],[0.3],[0.4],[0.5],[0.7],[0.78],[0.8],[0.9],[1.0]])
print predict_labels(test,train_model(values,lables))
