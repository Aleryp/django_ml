import datetime

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (180, 180)
NUM_CATEGORIES = 1000
BATCH_SIZE = 8
EPOCHS = 10

def get_base_model(input_shape):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 120

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return base_model

def create_model(input_shape):
    mobile_net = get_base_model(input_shape)

    inputs = tf.keras.Input(input_shape)

    x = mobile_net(inputs)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')(x)

    ret_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    ret_model.summary()
    return ret_model


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
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
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='training'
)
val_generator = datagen.flow_from_directory(
    dataset_dir,
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation'
)

model = create_model(train_generator.image_shape)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='src/model.keras',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
log_dir = "logs/based_on_model_partial_trainable/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=["accuracy"]
              )

model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=[
                                                                                    tensorboard_callback,
                                                                                    model_checkpoint_callback])

model.evaluate(val_generator)

model.save('scratch_model_based_on_model_partial_trainable.keras')
