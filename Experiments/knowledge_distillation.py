import os
import keras
import tensorflow as tf
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Conv2D, Flatten
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input


# Defining a custom model class
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def call(self, x, training=False):
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=True)
        return (teacher_predictions, student_predictions)

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        # Forward pass of teacher
        # teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            # student_predictions = self.student(x, training=True)
            teacher_predictions, student_predictions = self(x, training=True)
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha*student_loss
            loss += (1 - self.alpha)*distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dictionary of the performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss,
             "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        x, y = data
        teacher_predictions, student_predictions = self(x, training=False)

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dictionary of the performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


# This creates the teacher

# imports the mobilenet model and discards the last neuron layer.
base_model = MobileNet(
    weights='imagenet', include_top=False,
    input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Adding dense layers so that the model can learn more
# complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)
teacher = Model(inputs=base_model.input, outputs=preds)

for layer in teacher.layers[:20]:
    layer.trainable = False
for layer in teacher.layers[20:]:
    layer.trainable = True

# Create the student
student = Sequential()

# add model layers
student.add(
    Conv2D(64, kernel_size=3,
           activation='relu', input_shape=(224, 224, 3)))
student.add(Conv2D(64, kernel_size=3, activation='relu'))
student.add(Dropout(0.2))
student.add(Conv2D(32, kernel_size=3, activation='relu'))
student.add(Dropout(0.2))
student.add(Conv2D(32, kernel_size=3, activation='relu'))
student.add(Flatten())
student.add(Dense(2, activation='softmax'))

# Clone the student for later comparison
student_scratch = keras.models.clone_model(student)

# Setting up the teacher (original full neural network)
train_dir = r'path\to\training\data'
validation_dir = r'path\to\validation\data'
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

teacher.compile(
    optimizer=Adam(lr=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train and evaluate teacher on data.
teacher.fit(
    train_generator, validation_data=validation_generator,
    steps_per_epoch=int(num_training_images/batch_size),
    validation_steps=int(num_validation_images/batch_size),
    epochs=10)

teacher = keras.models.load_model('finalmodel.h5')
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=Adam(lr=0.00001),
    metrics=['accuracy'],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)
# Distill teacher to student
distiller.fit(
    train_generator, validation_data=validation_generator,
    steps_per_epoch=int(num_training_images/batch_size),
    validation_steps=int(num_validation_images/batch_size),
    epochs=50)
