# -*- coding: utf-8 -*-
"""Fine_tuned.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1p6flm4__DHgrjUT4PgS2wI4M8iqRleYH
"""

# Step 1: Install Kaggle library (if not already installed)
!pip install -q kaggle

# Step 2: Upload the Kaggle API key (kaggle.json)
from google.colab import files
files.upload()  # Run this and select the kaggle.json file you downloaded from Kaggle

# Step 3: Move kaggle.json to the right directory
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json  # Secure the API key file

# Step 4: List and download the dataset
# Replace "dataset-owner/dataset-name" with your dataset's path on Kaggle
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content/Brain
!kaggle datasets download -d stevenazy/liver-dataset -p /content/Liver
!kaggle datasets download -d omkarmanohardalvi/lungs-disease-dataset-4-types -p /content
!kaggle datasets download -d antonbudnychuk/hand-xray -p /content/Hand

import zipfile
import os

# Paths to each downloaded dataset
datasets = {
    "Brain": "/content/Brain/brain-tumor-mri-dataset.zip",
    "Liver": "/content/Liver/liver-dataset.zip",
    "Lung": "/content/lungs-disease-dataset-4-types.zip",
    "Hand": "/content/Hand/hand-xray.zip"
}

# Extract each dataset to its respective folder
for organ, path in datasets.items():
    with zipfile.ZipFile(path, 'r') as zip_ref:
        extraction_path = f"/content/{organ}"  # Set an extraction directory for each organ
        zip_ref.extractall(extraction_path)
        print(f"{organ} dataset extracted to {extraction_path}")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

tf.keras.mixed_precision.set_global_policy('float32')

def copy_files_maintaining_structure(src_dir, dest_parent_dir, organ_name):
    """
    Copies files from source directory to destination while flattening the structure
    but maintaining unique filenames
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get the subfolder name (if any) to preserve in filename
                rel_path = os.path.relpath(root, src_dir)
                if rel_path != '.':
                    # Create unique filename including subfolder information
                    new_filename = f"{rel_path.replace(os.sep, '_')}_{file}"
                else:
                    new_filename = file

                src_path = os.path.join(root, file)
                dest_dir = os.path.join(dest_parent_dir, organ_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, new_filename)
                shutil.copy2(src_path, dest_path)

# Define paths for each organ
base_dir = "/content"
train_data = {
    'Brain': os.path.join(base_dir, "Brain/Training"),
    'Hand': os.path.join(base_dir, "Hand/train/hand"),
    'Liver': os.path.join(base_dir, "Liver/liver/train"),
    'Lung': os.path.join(base_dir, "Lung/Lung Disease Dataset/train")
}

# Image parameters
image_size = (128, 128)
batch_size = 32

# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a temporary directory structure for the generator
import shutil
temp_train_dir = os.path.join(base_dir, 'temp_organized_data')
if os.path.exists(temp_train_dir):
    shutil.rmtree(temp_train_dir)
os.makedirs(temp_train_dir, exist_ok=True)

# Organize files into the required structure
for organ, src_dir in train_data.items():
    print(f"Processing {organ} images...")
    copy_files_maintaining_structure(src_dir, temp_train_dir, organ)

# Create generators
print("Creating data generators...")
train_generator = train_datagen.flow_from_directory(
    temp_train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    temp_train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Define callbacks
callbacks = [
    ModelCheckpoint(
        'best_organ_classifier.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# Define the CNN model with proper input layer
model = Sequential([
    Input(shape=(128, 128, 3)),  # Proper input layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes for 4 organs
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
try:
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
finally:
    # Clean up temporary directory
    shutil.rmtree(temp_train_dir)

# Save the final model
model.save('final_organ_classifier.keras')

# Step 1: Install Kaggle library (if not already installed)
!pip install -q kaggle

# Step 2: Upload the Kaggle API key (kaggle.json)
from google.colab import files
files.upload()  # Run this and select the kaggle.json file you downloaded from Kaggle

# Step 3: Move kaggle.json to the right directory
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json  # Secure the API key file

# Step 4: List and download the dataset
# Replace "dataset-owner/dataset-name" with your dataset's path on Kaggle
!kaggle datasets download -d jarvisgroot/brain-tumor-classification-mri-images -p /content/Brain
!kaggle datasets download -d priyamsaha17/liver-segmentation-dataset-2 -p /content/Liver
!kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset -p /content/Lung
!kaggle datasets download -d umeradnaan/x-ray-dection -p /content/Hand

import zipfile
import os

# Paths to each downloaded dataset
datasets = {
    "Brain": "/content/Brain/brain-tumor-classification-mri-images.zip",
    "Liver": "/content/Liver/liver-segmentation-dataset-2.zip",
    "Lung": "/content/Lung/tuberculosis-tb-chest-xray-dataset.zip",
    "Hand": "/content/Hand/x-ray-dection.zip"
}

# Extract each dataset to its respective folder
for organ, path in datasets.items():
    with zipfile.ZipFile(path, 'r') as zip_ref:
        extraction_path = f"/content/{organ}"  # Set an extraction directory for each organ
        zip_ref.extractall(extraction_path)
        print(f"{organ} dataset extracted to {extraction_path}")

import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import uuid

tf.keras.mixed_precision.set_global_policy('float32')

# Define base directory
base_dir = "/content"

# Define paths for each organ's training data
train_data = {
    'Brain': os.path.join(base_dir, "Brain/brain_tumor_mri/new_dataset/bt_images"),
    'Hand': os.path.join(base_dir, "Hand/Bone Fracture Dataset/training"),
    'Liver': os.path.join(base_dir, "Liver/images/images"),
    'Lung': os.path.join(base_dir, "Lung/TB_Chest_Radiography_Database")
}

def copy_files_with_unique_names(src_dir, dest_parent_dir, organ_name):
    """
    Copies image files from the source directory to the destination
    directory using unique filenames to prevent overwriting.
    """
    image_count = 0
    skipped_files = 0
    per_folder_counts = {}  # Track counts per folder

    for root, _, files in os.walk(src_dir):
        # Count total files per subfolder for Lung dataset tracking
        folder_name = os.path.relpath(root, src_dir)
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        per_folder_counts[folder_name] = len(valid_files)

        for file in valid_files:
            # Generate a unique filename to avoid overwriting
            unique_filename = f"{organ_name}_{uuid.uuid4().hex}_{file}"

            src_path = os.path.join(root, file)
            dest_dir = os.path.join(dest_parent_dir, organ_name)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, unique_filename)

            # Copy the file and count successful copies
            shutil.copy2(src_path, dest_path)
            image_count += 1

    print(f"\nTotal images copied for {organ_name}: {image_count}, Skipped non-image files: {skipped_files}")
    print(f"\nFile count per folder for {organ_name}: {per_folder_counts}")

# Clear the temporary directory before starting
temp_train_dir = os.path.join(base_dir, "temp_train")
if os.path.exists(temp_train_dir):
    shutil.rmtree(temp_train_dir)
os.makedirs(temp_train_dir, exist_ok=True)

# Print and count total images in each source directory before copying
for organ, path in train_data.items():
    total_files = sum(len(files) for _, _, files in os.walk(path) if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) for f in files))
    print(f"Starting copy for {organ}. Total files found in source: {total_files}")
    copy_files_with_unique_names(path, temp_train_dir, organ)

# Verify and print the total number of images copied for each organ
for organ in train_data.keys():
    organ_dir = os.path.join(temp_train_dir, organ)
    if os.path.exists(organ_dir):
        num_images = len([f for f in os.listdir(organ_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        print(f"Total {organ} images copied to {temp_train_dir}: {num_images}")
    else:
        print(f"No directory found for {organ} in {temp_train_dir}")

# Image parameters
image_size = (128, 128)
batch_size = 32

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping

# Step 1: Load the Pre-trained Model
model = load_model('/content/best_organ_classifier_copy.h5')  # Replace with your actual model path

# Step 2: Freeze Early Layers
# Freeze all layers except the last few (depending on your architecture)
for layer in model.layers[:-5]:  # Adjust the number of layers to freeze as needed
    layer.trainable = False

# Step 3: Prepare the Fine-Tuning Dataset
# Use ImageDataGenerator for augmenting your dataset to add variability
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Set aside 20% for validation
)

# Base directory for image data
base_dir = "/content"

# Define paths for each organ's training data
train_data = {
    'Brain': os.path.join(base_dir, "Brain/brain_tumor_mri/new_dataset/bt_images"),
    'Hand': os.path.join(base_dir, "Hand/Bone Fracture Dataset/training"),
    'Liver': os.path.join(base_dir, "Liver/images/images"),
    'Lung': os.path.join(base_dir, "Lung/TB_Chest_Radiography_Database")
}

# Combine the paths to create a single training directory structure
# Create a temporary directory structure for the ImageDataGenerator
temp_train_dir = os.path.join(base_dir, "temp_train")

# Create a directory for each organ in the temporary train directory
for organ, path in train_data.items():
    os.makedirs(os.path.join(temp_train_dir, organ), exist_ok=True)

    # Copy images from each organ's path to the temporary directory
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other extensions as necessary
            src = os.path.join(path, filename)
            dst = os.path.join(temp_train_dir, organ, filename)
            if os.path.isfile(src):
                shutil.copy(src, dst)

# Prepare the train and validation generators
train_generator = train_datagen.flow_from_directory(
    temp_train_dir,
    target_size=(128, 128),  # Use the same size as your original training
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    temp_train_dir,  # Same path as train data
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Step 4: Compile the Model with a Lower Learning Rate
model.compile(optimizer=Adam(learning_rate=1e-5),  # Use a lower learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Fine-Tune the Model
# Set up EarlyStopping to monitor validation loss
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop training if no improvement in validation loss for 5 epochs
    restore_best_weights=True  # Restore the model weights from the epoch with the best validation loss
)

# Train on the augmented dataset to help it generalize better
history = model.fit(
    train_generator,
    epochs=50,  # You can increase the epochs for early stopping to take effect
    validation_data=validation_generator,
    callbacks=[early_stopping]  # Include the EarlyStopping callback
)

# Save the model to Google Drive
model.save('/content/drive/MyDrive/fine_tuned_model.h5')


# Optional: Clean up temporary directory after training
shutil.rmtree(temp_train_dir)  # Remove the temporary directory

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the fine-tuned model
model_path = '/content/drive/MyDrive/fine_tuned_model.h5'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Define the path to the directory containing test images
test_image_dir = '/content/test_images'  # Change this to your test images directory

# Image preprocessing parameters
image_size = (128, 128)  # Adjust based on model's expected input size

# Define a dictionary to map class indices to organ names
class_labels = {
    0: 'Brain',
    1: 'Hand',
    2: 'Liver',
    3: 'Lung'
}

def load_and_preprocess_image(img_path):
    """
    Loads and preprocesses an image to be compatible with the model's input requirements.
    """
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Scale pixel values
    return img_array

# Function to predict and display results
def predict_and_display_results(model, test_image_dir):
    """
    Loads each image from the test directory, processes it, makes a prediction,
    and displays the result along with probabilities for each class.
    """
    for img_file in os.listdir(test_image_dir):
        img_path = os.path.join(test_image_dir, img_file)

        # Preprocess the image
        img_array = load_and_preprocess_image(img_path)

        # Make a prediction
        prediction = model.predict(img_array)[0]

        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_labels[predicted_class_index]
        confidence = prediction[predicted_class_index]

        # Display the image
        plt.imshow(image.load_img(img_path))
        plt.axis('off')

        # Display the predicted organ with confidence
        plt.title(f"Prediction: {predicted_class_name} (Confidence: {confidence:.2f})")
        plt.show()

        # Print probabilities for all classes
        print(f"\nProbabilities for {img_file}:")
        for i, prob in enumerate(prediction):
            print(f"{class_labels[i]}: {prob:.2%}")

# Run predictions on test images
predict_and_display_results(model, test_image_dir)