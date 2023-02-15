import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

import distortion_aware_ops as distortion

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import cv2

EPOCHS = 5

# Show a result of the activation map.
def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    filters = np.transpose(filters, [0, 3, 1, 2])
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], interpolation='nearest')
    plt.show()

if __name__=="__main__":

    train_ds, test_ds = tfds.load('horses_or_humans', split=['train', 'test'], shuffle_files=False)

    def normalization_layer(example):
        
        image = example["image"]
        label = example["label"]
        
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255.)

        return image, label

    b_size = 16
    train_ds = train_ds.map(normalization_layer, num_parallel_calls=tf.data.AUTOTUNE).shuffle(500).batch(b_size).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(normalization_layer, num_parallel_calls=tf.data.AUTOTUNE).batch(b_size).prefetch(tf.data.AUTOTUNE)
    
    class MyModel(Model):
        def __init__(self, skydome="skydome"):
            super(MyModel, self).__init__()
            self.conv1 = distortion.conv2d(16, kernel_size=3, strides=1, dilation_rate=1, skydome=skydome)  # out 24
            self.pool1 = layers.MaxPool2D()
            self.conv2 = distortion.conv2d(32, kernel_size=3, strides=1, dilation_rate=1, skydome=skydome)  # out 24Conv2D(32, 3, activation='relu')
            self.pool2 = layers.MaxPool2D()
            self.conv3 = distortion.conv2d(32, kernel_size=3, strides=1, dilation_rate=1, skydome=skydome)  # out 24Conv2D(32, 3, activation='relu')
            self.pool3 = layers.MaxPool2D()
            self.conv4 = distortion.conv2d(32, kernel_size=3, strides=1, dilation_rate=1, skydome=skydome)  # out 24Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10)

        def call(self, x):
            conv1 = self.conv1(x)
            x = self.pool1(conv1)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x), conv1

    # Create an instance of the model
    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions, conv1 = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

        return conv1

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, conv1 = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
        
        return conv1

    acc,val_acc,loss, val_loss = [], [], [], []
    
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for image, label in train_ds:
            _ = train_step(image, label)
            
        for image, label in test_ds:
            conv1 = test_step(image, label)
        
        # filter_show(conv1)
        
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )

        acc.append(train_accuracy.result() * 100)
        val_acc.append(test_accuracy.result() * 100)

        loss.append(train_loss.result())
        val_loss.append(test_loss.result())

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()