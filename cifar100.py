
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *
from light_dense import LightDense
from light_conv2d import LightConv2D

mnist = tf.keras.datasets.cifar100

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

tf.random.set_seed(42)
np.random.seed(42)

def conv_block(x, f, k=1):
    x_init = x
    x = LightConv2D(f, (3, 3), padding="SAME", k=k)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = LightConv2D(f, (3, 3), padding="SAME", k=k)(x)
    x = BatchNormalization()(x)

    s = LightConv2D(f, (1, 1), padding="SAME", k=k)(x_init)
    s = BatchNormalization()(s)

    x = Add()([x, s])
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)
    return x

def conv_block2(x, f, k=1):
    x_init = x
    x = Conv2D(f, (3, 3), padding="SAME")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f, (3, 3), padding="SAME")(x)
    x = BatchNormalization()(x)

    s = Conv2D(f, (1, 1), padding="SAME")(x_init)
    s = BatchNormalization()(s)

    x = Add()([x, s])
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)
    return x

def conv_block3(x, f, k=1):
    x_init = x
    x = SeparableConv2D(f, (3, 3), padding="SAME")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(f, (3, 3), padding="SAME")(x)
    x = BatchNormalization()(x)

    s = SeparableConv2D(f, (1, 1), padding="SAME")(x_init)
    s = BatchNormalization()(s)

    x = Add()([x, s])
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)
    return x

inputs = Input((32, 32, 3))
x = inputs

x = conv_block3(x, 16)
x = conv_block3(x, 32)
x = conv_block3(x, 64)
x = GlobalAveragePooling2D()(x)

x = Dense(100)(x)
x = Activation('softmax')(x)

model = tf.keras.models.Model(inputs, x)
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint("files/model_cifar10.h5", monitor='val_loss', verbose=1, save_weights_only=False, mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1),
    CSVLogger("files/data_cifar10.csv"),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1)
]

model.fit(x_train, y_train,
    epochs=20,
    shuffle=False,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    batch_size=64
    )


model.load_weights("files/model_cifar10.h5")
model.evaluate(x_test, y_test, batch_size=128, verbose=1)
