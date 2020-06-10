from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--data_dir", default='cats_and_dogs_small')
parser.add_argument("--log_dir", default='logs')
args = parser.parse_args()

train_dir = os.path.join(args.data_dir, 'train')
validation_dir = os.path.join(args.data_dir, 'validation')
test_dir = os.path.join(args.data_dir, 'test')
log_dir = os.path.join(args.log_dir, args.label)
save_path = os.path.join(log_dir, 'model.h5')

# Create ImageDataGenerator for training
train_datagen = ImageDataGenerator(
    # Normalize inputs to the network
    preprocessing_function=preprocess_input,
    # Use data augmentation during training
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
# Skip data augmentation during testing
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Read training dataset
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        # Read 32 images at once
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Read validation dataset
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# Load pre-trained model without classification layers (include_top=False)
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

# Add our own classification layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Switch training of pre-trained layers off
conv_base.trainable = False
model.summary()

# Use binary cross-entropy together with sigmoid activation
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.00005),
              metrics=['acc'])

# Train the classification layer
model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=[
        callbacks.TensorBoard(log_dir=log_dir),
        callbacks.ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True)
      ])

# Freeze all layers up to a specific one
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable

# Need to recompile after changing trainable layers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.00005),
              metrics=['acc'])

model.summary()

# Fine-tune also convolutional layers
model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50,
      initial_epoch=10,
      callbacks=[
        callbacks.TensorBoard(log_dir=log_dir),
        callbacks.ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True)
      ])

# Test the model on test set
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
