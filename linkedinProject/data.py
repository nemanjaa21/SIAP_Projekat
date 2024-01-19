import pandas as pd
import numpy as np
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences


def import_data():
    np.random.seed(123)
    data = pd.read_csv('../dataset/archive/job_postings.csv', index_col=False)
    # subset the data
    rand_job_ids = np.random.choice(data['job_id'].unique(),
                                     size=int(len(data['job_id'].unique()) * 0.01),
                                     replace=False)

    data = data.loc[data['job_id'].isin(rand_job_ids)]
    data = data.drop_duplicates(subset='job_id', keep="first")

    #Handle NaN values
    data = data.fillna(value=0)
    data['views'] = data['views'].fillna(value=0)  # Replace NaN in the 'views' column with 0
    data['views'] = data['views'].where(data['views'] > 200000, 1)
    data['views'] = data['views'].where(data['views'] < 200000, 0)
    print(data)
    return data


def split_data(jobs):
    # split the train/test split by the latest rating
    train_jobs = jobs.sample(frac=0.8, random_state=200)
    test_jobs = jobs.drop(train_jobs.index)

    return train_jobs, test_jobs


def split_input_output(train, test):
    x_train = train.drop(['views'], axis=1)
    y_train = train[['views']]

    x_test = test.drop(['views'], axis=1)
    y_test = test[['views']]

    return x_train, y_train, x_test, y_test