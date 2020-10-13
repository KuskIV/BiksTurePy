import math
import  tensorflow as tf
from tensorflow.keras import datasets, layers, models

def store_model(model, path):
    tf.keras.models.save_model(
        model,
        filepath= path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )

def train_model(train_images, train_labels, test_images, test_labels, SAVE_LOAD_PATH, save_model = True):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(43))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    for i in range(10):
        total_examples = train_images.shape[0]
        subset = math.floor(total_examples * 0.1)
        start = math.floor(i * total_examples)
        end = start + subset
        print(f'total_examples = {total_examples} | subset = {subset} | start = {start} | end = {end}')
        history = model.fit(train_images[start:end], train_labels[start:end], epochs=10,
                        validation_data=(test_images, test_labels))

    if save_model:
        store_model(model, SAVE_LOAD_PATH)

def flatten_and_dense(model):
    """Returns a model flattened and densed to 43 categories of prediction"""
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(43))
    return model

