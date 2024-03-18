import pandas as pd
import numpy as np
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder


def import_data():
    np.random.seed(123)
    data = pd.read_csv('../dataset/archive/job_postings.csv', index_col=False)
    eda(data)
    # subset the data
    rand_job_ids = np.random.choice(data['job_id'].unique(),
                                     size=int(len(data['job_id'].unique()) * 0.01),
                                     replace=False)

    data = data.loc[data['job_id'].isin(rand_job_ids)]
    data = data.drop_duplicates(subset='job_id', keep="first")


    print(data)
    return data

def import_data2():
    np.random.seed(123)
    data = pd.read_csv('../dataset/archive2/job_postings.csv', index_col=False)
    # subset the data
    rand_job_ids = np.random.choice(data['job_id'].unique(),
                                     size=int(len(data['job_id'].unique()) * 0.01),
                                     replace=False)

    data = data.loc[data['job_id'].isin(rand_job_ids)]
    data = data.drop_duplicates(subset='job_id', keep="first")


    print(data)
    return data


def split_data(jobs):
    # split the train/test split by the latest rating
    train_jobs = jobs.sample(frac=0.7, random_state=200)
    test_jobs = jobs.drop(train_jobs.index)

    return train_jobs, test_jobs

def eda(df):
    # Display basic information about the dataset
    print("Dataset Info:")
    print(df.info())

    # Display summary statistics for numerical columns
    print("\nSummary Statistics for Numerical Columns:")
    print(df.describe())

    # Display summary statistics for categorical columns
    print("\nSummary Statistics for Categorical Columns:")
    print(df.describe(include=['object']))

    print("\nVisualize the distribution of numerical columns:")
    # Visualize the distribution of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sb.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    # Pairplot for numerical columns
    plt.figure(figsize=(12, 10))
    sb.pairplot(df[numerical_cols])
    plt.title('Pairplot of Numerical Columns')
    plt.show()

    # Correlation heatmap for numerical columns
    plt.figure(figsize=(10, 8))
    sb.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Columns')
    plt.show()
def split_input_output(train, test):
    x_train = train.drop(['views'], axis=1)
    y_train = train[['views']]

    x_test = test.drop(['views'], axis=1)
    y_test = test[['views']]

    return x_train, y_train, x_test, y_test

def preprocessing_data(data):
    # Handle NaN values
    data['views'] = data['views'].fillna(data['views'].median())
    data['applies'] = data['applies'].fillna(data['applies'].median())

    # Convert categorical variables to numerical values
    label_encoder = LabelEncoder()
    data['title'] = label_encoder.fit_transform(data['title'])
    data['description'] = label_encoder.fit_transform(data['description'])
    data['location'] = label_encoder.fit_transform(data['location'])

    data['views'] = data['views'].fillna(value=0)
    data['applies'] = data['applies'].fillna(value=0)
    data['views'] = data['views'].where(data['views'] > 100, 1)
    data['views'] = data['views'].where(data['views'] < 100, 0)
    data['applies'] = data['applies'].where(data['applies'] > 40, 1)
    data['applies'] = data['applies'].where(data['applies'] < 40, 0)
    print(data)

    return data