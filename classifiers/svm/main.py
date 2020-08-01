import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump

import os,sys
sys.path.append(os.path.abspath('../../utils'))
from compute_metrics import compute_metrics

# load feature vectors and labels

X = np.load('../features/X.npy')
with open('../features/labels.pkl','rb') as f:
    y = pickle.load(f)

# split data

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y)

# intiialize the SVM

clf = SVC(C = 200.0,
          kernel = 'rbf',
          gamma = 0.00001,
          class_weight = 'balanced',
          random_state = 42)

# train the SVM

clf.fit(X_train, y_train)

# compute predictions on test data

preds = clf.predict(X_test)

# compute prediction metrics and show them

metrics = compute_metrics(y_true = y_test,
                          y_pred = preds)

print('\nPrediction metrics:')
print('\nConfusion Matrix:\n{}\n'.format(metrics['cfm']))
print('Sensitivity/Recall: {:.3f}%'.format(metrics['sens']*100))
print('Specificity: {:.3f}%'.format(metrics['spec']*100))
print('Precision/PPV: {:.3f}%'.format(metrics['PPV']*100))
print('NPV: {:.3f}%'.format(metrics['NPV']*100))
print('Accuracy: {:.3f}%'.format(metrics['acc']*100))
print('Balanced Accuracy: {:.3f}%'.format(metrics['bal_acc']*100))
print('Matthews correlation coefficient: {:.3f}'.format(metrics['MCC']))

# save the SVM

dump(clf,'svm_param.joblib')
