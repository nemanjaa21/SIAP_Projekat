from random import randint

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from data import *
#Support Vector Machine classifier for predicting popularity based on views
if __name__ == '__main__':
    train, test = split_data(preprocessing_data(import_data()).drop(['job_id','company_id', 'title', 'description','max_salary',
                                                                     'max_salary','med_salary','min_salary','pay_period',
                                                                     'formatted_work_type','location','original_listed_time',
                                                                     'remote_allowed', 'job_posting_url', 'application_url',
                                                                     'application_type','expiry','closed_time','formatted_experience_level',
                                                                     'skills_desc','listed_time','posting_domain','sponsored',
                                                                     'work_type', 'currency','compensation_type', 'scraped'
                                                                    ], axis=1))
    x_train, y_train, x_test, y_test = split_input_output(train, test)
    # Convert DataFrames to 1-dimensional arrays
    y_train = np.ravel(y_train.values)
    y_test = np.ravel(y_test.values)
    clf = svm.SVC(kernel='rbf', C=100, gamma=0.1)  #gamma 0.1
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("AUC:", metrics.roc_auc_score(y_test, y_pred))
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    # print f1 score
    print("F1:", metrics.f1_score(y_test, y_pred))
   # wrong_predictions = test[y_pred != y_test.to_numpy().flatten()]
    correct_indices = np.where(y_pred == y_test)[0]
    correct_predictions = test.iloc[correct_indices]
    wrong_indices = np.where(y_pred != y_test)[0]
    wrong_predictions = test.iloc[wrong_indices]
    plt.hist([wrong_predictions.loc[wrong_predictions['views'] == 0, 'applies'],
                correct_predictions.loc[correct_predictions['views'] == 1, 'applies']],
                stacked=False,
    label=['Unpopular', 'Popular'],
    edgecolor='white')
    plt.legend()
    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #
    # # Use random search to find the best hyperparameters
    # rand_search = RandomizedSearchCV(clf,
    #                                  param_distributions=param_grid,
    #                                  refit=True,
    #                                  verbose=2)
    #
    # # Fit the random search object to the data
    # rand_search.fit(x_train, y_train)
    #
    # # Create a variable for the best model
    # best_svm = rand_search.best_estimator_
    #
    # # Print the best hyperparameters
    # print('Best hyperparameters:', rand_search.best_params_)