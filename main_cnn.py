import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

import distortion_filter as f

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import cv2

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

# def filter_show(img, filters, nx=8, margin=3, scale=10):
#     """
#     c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
#     """
#     filters = np.transpose(filters, [0, 3, 1, 2])
#     FN, C, FH, FW = filters.shape
#     ny = int(np.ceil(FN / nx))

#     fig = plt.figure()
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

#     img = np.zeros_like(img[0])
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     print(np.shape(img))
#     for i in range(FN):
        
#         ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
#         h,w = np.shape(filters[i, 0])
#         print(filters[i, 0])
        
#         for y in range(h):
#             for x in range(w):
                
#                 # cv2.circle(img, filters[i, 0, y, x], 1, (0,0,255))
#                 # cv2.imshow("img",img)
#                 # cv2.waitKey(0)
#                 plt.imshow(img, interpolation="nearest")
#                 plt.show()
    # plt.show()

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
        def __init__(self):
            super(MyModel, self).__init__()
            # self.conv1 = Conv2D(16, 3, activation='relu')
            self.conv1 = f.DistortionConvLayer(16, [3, 3])  # out 24
            self.pool1 = layers.MaxPool2D()
            self.conv2 = Conv2D(32, 3, activation='relu')
            self.pool2 = layers.MaxPool2D()
            self.conv3 = Conv2D(32, 3, activation='relu')
            self.pool3 = layers.MaxPool2D()
            self.conv4 = Conv2D(32, 3, activation='relu')
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

    EPOCHS = 5

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
        
        filter_show(conv1.numpy())
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
    ####################################

    # acc = train_accuracy.result()
    # val_acc = test_accuracy.result()

    # loss= train_loss.result()
    # val_loss=test_loss.result()

    # epochs_range = range(EPOCHS)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()