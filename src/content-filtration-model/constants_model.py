import os

# Defining our paths for training and validation steps
train_dir = r'path\to\training\data'
validation_dir = r'path\to\validation\data'

# Target image dimensions
dimension1, dimension2, dimension3 = 224, 224, 3
# For training
batch_size = 30

''' For "content-filtration-model.py" '''

# callback Stop_Validation
stopping_acc = 0.9600
stopping_loss = 0.1000

# callback callback_reduce_lr
monitor = 'val_loss'
factor = 0.2
patience = 5
min_lr = 0.0001

# For Adam optimizer
learning_rate = 0.0001
loss = 'categorical_crossentropy'
metrics = ['accuracy']

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
