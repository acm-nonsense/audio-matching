import numpy as np
import os
import sys
from sknn.mlp import Classifier, Convolution, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def process_files(output_root):
    files = os.listdir(os.path.join(output_root,'npz'))
    with np.load(os.path.join(output_root,'npz',files[0])) as first_ssm_file:
        first_ssm = first_ssm_file['arr_0']
        values = np.ndarray((len(files),first_ssm.shape[0],first_ssm.shape[1]))
        labels = np.array(["" for _ in range(len(files))],dtype=object)
        for i in range(len(files)):
            npz_type = files[i][-8:-4]
            if npz_type == 'assm':
                with np.load(os.path.join(output_root,'npz',files[i])) as ssm_file:
                    ssm = ssm_file['arr_0']
                    print("{}: {}".format(files[i],ssm.shape))
                    values[i] = ssm
                    labels[i] = files[i][:-15]
    return (values,labels)

def train_model(values,labels):
    model = Classifier(
	layers=[
		Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
		Layer("Softmax")
	],
	learning_rate=0.02,
	n_iter=5)
    model.fit(values, labels)
    return model

def predict_labels(values,model):
    return model.predict(values)

def main():
    values,labels = process_files(sys.argv[1])
    v_train, v_test, l_train, l_test = train_test_split(values, labels, test_size=0.2, random_state=42)
    print(labels)
    print(v_train.shape)
    print(v_test.shape)
    print(l_train.shape)
    print(l_test.shape)
    l_pred =  predict_labels(v_test,train_model(v_train,l_train))
    print accuracy_score(l_test,l_pred)
    cm =  confusion_matrix(l_test,l_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.show()

if __name__ == "__main__":
    main()
