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


if _name_ == '_main_':
    train, test = split_data(import_data())
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    num_locations = 5  # Number of unique issue tags
    num_words = 10000  # Size of vocabulary obtained when preprocessing text data
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
    padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_train['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    encoded_location = [one_hot(d, num_words) for d in x_train['location']]
    padded_locations = pad_sequences(encoded_location, maxlen=5, padding='post')
    title_data = padded_titles
    description_data = padded_description
    location_data = padded_locations

    dept_targets = y_train

    model.fit(
        {"title": title_data, "description": description_data, "location": location_data},
        {"views": dept_targets},
        epochs=5,
        batch_size=32,
    )
