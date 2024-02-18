# This is a sample Python script.

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from data import *


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    train, test = split_data(import_data())
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    num_locations = 30  # Number of unique issue tags
    num_words = 100000  # Size of vocabulary obtained when preprocessing text data
    num_pop_levels = 4  # Number of departments for predictions

    title_input = keras.Input(shape=(None,), name="title")
    description_input = keras.Input(shape=(None,), name="description")
    location_input = keras.Input(shape=(num_locations,), name="location")

    title_features = layers.Embedding(num_words, 64)(title_input)
    description_features = layers.Embedding(num_words, 64)(description_input)

    title_features = layers.LSTM(128)(title_features)
    description_features = layers.LSTM(32)(description_features)

    x = layers.concatenate([title_features, description_features, location_input])

    popularity_pred = layers.Dense(1, name="views")(x)
    print(popularity_pred)

    model = keras.Model(
        inputs=[title_input, description_input, location_input],
        outputs=[popularity_pred],
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[
            keras.losses.BinaryCrossentropy(from_logits=True),
            keras.losses.CategoricalCrossentropy(from_logits=True),
        ],
        loss_weights=[1.0, 0.2],
        metrics=[keras.metrics.SparseCategoricalAccuracy(), "accuracy"],
    )

    encoded_titles = [one_hot(d, num_words) for d in x_train['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=30, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_train['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=30, padding='post')
    encoded_location = [one_hot(d, num_words) for d in x_train['location']]
    padded_locations = pad_sequences(encoded_location, maxlen=30, padding='post')
    title_data = padded_titles
    description_data = padded_description
    location_data = padded_locations

    dept_targets = y_train

    model.fit(
        {"title": title_data, "description": description_data, "location": location_data},
        {"views": dept_targets},
        epochs=7,
        batch_size=32,
    )

    encoded_titles = [one_hot(d, num_words) for d in x_test['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=30, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_test['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=30, padding='post')
    encoded_locations = [one_hot(d, num_words) for d in x_test['location']]
    padded_locations = pad_sequences(encoded_locations, maxlen=30, padding='post')
    title_data = padded_titles
    description_data = padded_description
    location_data = padded_locations

    print("Length of title_data:", len(padded_titles))
    print("Length of description_data:", len(padded_description))
    print("Length of location_data:", len(padded_locations))
    print("Length of y_test:", len(y_test))

    test_scores = model.evaluate([title_data, description_data, location_data], y_test, verbose=2)

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print("Test sparse accuracy:", test_scores[2])
    print_hi('PyCharm')
