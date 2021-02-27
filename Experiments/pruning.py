import os
import tempfile
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from tensorflow_model_optimization.sparsity import keras as sparsity

model = tf.keras.models.load_model('keras_model.h5')

# Storing paths and their constituents

train_dir = r'path\to\train\folder'
validation_dir = r'path\to\validation\folder'

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

batch_size = 30

# Specifying the data generators for training
# and validation steps
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), color_mode='rgb',
    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, batch_size=batch_size, color_mode='rgb',
    class_mode='categorical', target_size=(224, 224),
    shuffle=True)

# Specifying training bounds
epochs = 5
end_step = np.ceil(1.0 * num_training_images / 30).astype(np.int32)*epochs
print(end_step)

new_pruning_parameters = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.50, final_sparsity=0.90,
        begin_step=0, end_step=end_step, frequency=100)
}

new_pruned_model = sparsity.prune_low_magnitude(
    model, **new_pruning_parameters)
# new_pruned_model.summary()

new_pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Adding early stopping and reduce learning rate
# callbacks to be invoked after every epoch
callback_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2)
callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=2,
    min_lr=0.0001, cooldown=1)

logdir = tempfile.mkdtemp()

callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
    callback_early_stopping,
    callback_reduce_lr
]

# Initiating the pruning schedule
new_pruned_model.fit(
    train_generator, validation_data=validation_generator,
    steps_per_epoch=int(num_training_images/batch_size),
    validation_steps=int(num_validation_images/batch_size),
    epochs=epochs, callbacks=callbacks)
