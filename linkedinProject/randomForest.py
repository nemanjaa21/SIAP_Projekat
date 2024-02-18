from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    roc_auc_score, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from data import *

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
    y_train = np.ravel(y_train.values)
    y_test = np.ravel(y_test.values)
    rf = RandomForestClassifier(n_estimators=379, max_depth=2, random_state=0)  # max_depth = 2 n_estimators = 379
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("ROC:", auc_roc)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    # print f1 score
    print("F1:", metrics.f1_score(y_test, y_pred))
    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    # param_dist = {'n_estimators': randint(50, 500),
    #              'max_depth': randint(1, 20)}
    # Use random search to find the best hyperparameters
    # rand_search = RandomizedSearchCV(rf,
    #                                  param_distributions=param_dist,
    #                                 n_iter=5,
    #                                 cv=5)
    # Fit the random search object to the data
    # rand_search.fit(x_train, y_train)
    # Create a variable for the best model
    # best_rf = rand_search.best_estimator_
    # Print the best hyperparameters
    # print('Best hyperparameters:', rand_search.best_params_)