# Installing dependencies
import os
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

# Defining our paths for training and validation steps
train_dir = r'path\to\training\data'
validation_dir = r'path\to\validation\data'

# Removing the last layer of MobileNet and adding the rest
base_model = MobileNet(weights='imagenet',
                       include_top=False, input_shape=(224, 224, 3))

# Defining a custom secondary architecture to existing MobileNet model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

# Freezing the first 20 layers and setting the rest as trainable
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

# Defining a function that stops training once
# the model reaches a certain threshold


class Stop_Validation(tf.keras.callbacks.Callback):

    def __init__(self, acc_threshold, loss_threshold):
        super(Stop_Validation, self).__init__()
        self.acc_threshold = acc_threshold
        self.loss_threshold = loss_threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        val_loss = logs["val_loss"]
        if val_acc >= self.acc_threshold and val_loss <= self.loss_threshold:
            self.model.stop_training = True


callback_stop_training = Stop_Validation(
    acc_threshold=0.9616, loss_threshold=0.1111)

# Setting a callback to reduce learning rate based on validation loss
callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model.compile(optimizer=Adam(lr=0.00001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Setting train and validation data generators for training and validation
batch_size = 30

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), color_mode='rgb',
    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1.0/255.)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, batch_size=batch_size, class_mode='categorical',
    target_size=(224, 224))

unsafe_images_train = [fn for fn in os.listdir(
    r'path\to\train\unsafe_images') if fn.endswith('.extension')]
safe_images_train = [fn for fn in os.listdir(
    r'path\to\train\safe_images') if fn.endswith('.extension')]
unsafe_images_validation = [fn for fn in os.listdir(
    r'path\to\validation\unsafe_images') if fn.endswith('.extension')]
safe_images_validation = [fn for fn in os.listdir(
    r'path\to\validation\safe_images') if fn.endswith('.extension')]

num_training_images = len(unsafe_images_train+safe_images_train)
num_validation_images = len(unsafe_images_validation+safe_images_validation)

# Fitting the model
history = model.fit(
    train_generator, validation_data=validation_generator,
    # num_training_images = total number of training images
    steps_per_epoch=int(num_training_images/batch_size),
    # num_validation_images = total number of validation images
    validation_steps=int(num_validation_images/batch_size),
    epochs=10, callbacks=[callback_reduce_lr, callback_stop_training])

model.save('name_of_model.h5')
