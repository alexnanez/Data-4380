# Importing necessary modules
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

def load_data():
    # Generating a dataset
    image_size = (224, 224)

    batch_size = 10

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "train",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    return train_ds, val_ds

def aug_data(train_ds, val_ds):

    # Using image data augmentation
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(.5,.2)
    ]


    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    
    return train_ds, val_ds