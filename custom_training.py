import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
IMAGE_SIZE = (224, 224)
NUM_CATEGORIES = 1000
BATCH_SIZE = 16
EPOCHS = 10

# Custom Activation Function (Leaky ReLU)
class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        super(LeakyReLU, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.where(inputs > 0, inputs, inputs * self.alpha)

    def get_config(self):
        config = super(LeakyReLU, self).get_config()
        config.update({'alpha': self.alpha})
        return config


# Custom BatchNorm (Trainable Momentum)
class TrainableBatchNorm(tf.keras.layers.BatchNormalization):
    def __init__(self, epsilon=1e-5, momentum=0.1, trainable_momentum=True, **kwargs):
        super(TrainableBatchNorm, self).__init__(epsilon=epsilon, momentum=momentum, **kwargs)
        self.trainable_momentum = trainable_momentum

    def build(self, input_shape):
        super(TrainableBatchNorm, self).build(input_shape)
        if self.trainable_momentum:
            self.momentum = self.add_weight(name='momentum', shape=(), initializer='zeros', trainable=False)

    def call(self, inputs, training=None, mask=None):
        return super(TrainableBatchNorm, self).call(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(TrainableBatchNorm, self).get_config()
        config.update({'trainable_momentum': self.trainable_momentum})
        config.update({'momentum': self.momentum.numpy()})  # Convert momentum to numpy array
        return config

def create_model(input_shape):
    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = LeakyReLU()(x)
    x = TrainableBatchNorm()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)


    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = LeakyReLU()(x)
    x = TrainableBatchNorm()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)


    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = LeakyReLU()(x)
    x = TrainableBatchNorm()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)


    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')(x)

    ret_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    ret_model.summary()
    return ret_model


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

dataset_dir = "categorical"

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='training'
)
val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation'
)

model = create_model(train_generator.image_shape)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='scratch_model_SGD_opt.keras',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
log_dir = "logs/sgd_optimizer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=["accuracy"]
              )

model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=[early_stopping,
                                                                                    tensorboard_callback,
                                                                                    model_checkpoint_callback])

model.evaluate(val_generator)

model.save('scratch_model_SGD_opt.keras')
