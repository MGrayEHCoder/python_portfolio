# Big Mike Jak and Alex J
# 9/30/25
# NN project bio insp.

import numpy as np
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

# === Load Data ===
data_dir = pathlib.Path('TrainingData')

batch_size = 43
img_height = 128
img_width = 128

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=124,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=124,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# This adds robustness to the network, zoom made it a little too robust.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    # tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])


model = tf.keras.Sequential([
    data_augmentation,

    # This normalizes the data
    tf.keras.layers.Rescaling(1./255),
    
    # All of these layers are two dimensional convolutions
    tf.keras.layers.Conv2D(15, (3,3), activation='linear', padding='same'),
    # tf.keras.layers.DepthwiseConv2D(3, activation='linear'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, (3,3), activation='linear', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (4,4), activation='linear', padding='same'),
    # tf.keras.layers.DepthwiseConv2D(3, activation='linear'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (4,4), activation='linear', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, (4,4), activation='linear', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(512, (5,5), activation='linear', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    # This is basically a fancy way to flatten the 2D data into 1D
    tf.keras.layers.GlobalAveragePooling2D(), 

    #This drops some of the data to prevent over training
    tf.keras.layers.Dropout(0.4),

    #These are just your standard densely connected layers
    tf.keras.layers.Dense(512, activation='selu'),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(256, activation = 'selu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation = 'selu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation = 'selu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(num_classes, activation="softmax")
])
# model = tf.keras.models.load_model('C:/Users/micha/OneDrive - murraystate.edu/BioInspired-Michaels_Laptop/best_model.keras')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

#This will stop the training early if the validation accuracy doesn't improve in 30 epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=30,
    restore_best_weights=True
)

#This saves the model with the highest validation accuracy for later use
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "Model.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

#Training the model for 300 epochs
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=600,

    #you would add in the early_stop here to add it in
    callbacks=[checkpoint]
)

#This recalls the metrics of the system
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#This gets and prints the best value for all of the metrics
best_val_acc = max(history.history["val_accuracy"])
print(f"Best validation accuracy: {best_val_acc:.4f}")
best_train_acc = max(history.history["accuracy"])
print(f"Best training accuracy: {best_train_acc:.4f}")
best_val_loss = min(history.history["val_loss"])
print(f"Best validation loss: {best_val_loss:.4f}")
best_loss = min(history.history["loss"])
print(f"Best training loss: {best_loss:.4f}")

#This just gets how many epochs we went through if we implemented the early_stop
epochs_range = range(len(acc))

#Creates the figure
plt.figure(figsize=(12, 6))

#Plots the Accuracy vs Training epoch
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training vs Validation Accuracy")

#Plots the Loss vs Training epoch
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training vs Validation Loss")

plt.savefig('Network_Performance.png')
plt.show()