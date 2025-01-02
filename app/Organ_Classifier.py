import sys
import json
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QFrame, QHBoxLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage,QIcon
import cv2
from tensorflow import keras
import tensorflow as tf



class OrganClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Organ Classifier")
        self.setMinimumSize(800, 600)
        app_icon = QIcon("../assets/logo.png")
        self.setWindowIcon(app_icon)
        self.model = None
        self.class_labels = None  # Will be loaded with the model
        self.input_shape = None
        self.CONFIDENCE_THRESHOLD = 40.0  # Minimum confidence threshold

        # Initialize UI
        self.setup_ui()

    def setup_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create button container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)

        # Create load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        button_layout.addWidget(self.load_model_btn)

        # Create upload image button (initially disabled)
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setEnabled(False)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_btn)

        # Add button container to main layout
        layout.addWidget(button_container)

        # Create image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        layout.addWidget(self.image_label)

        # Create result label
        self.result_label = QLabel("Please load the organ classification model")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 10px;
                margin: 10px;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.result_label)

    def load_model_dialog(self):
        model_file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Keras Model (*.keras);;SavedModel Directory (*)"
        )

        if model_file_path:
            try:
                # First try loading labels
                labels_file_path = Path(model_file_path).with_suffix('.json')
                if not labels_file_path.exists():
                    self.show_error(
                        "No labels file found. Please ensure there's a JSON file with class labels in the same directory as the model.")
                    return

                with open(labels_file_path, 'r') as f:
                    self.class_labels = json.load(f)

                # Try multiple loading approaches with better error handling
                loading_approaches = [
                    # Approach 1: Standard Keras load
                    lambda: keras.models.load_model(model_file_path, compile=False),

                    # Approach 2: Load with custom object scope
                    lambda: keras.models.load_model(model_file_path, compile=False,
                                                    custom_objects={'CustomLayer': keras.layers.Layer}),

                    # Approach 3: Load as SavedModel
                    lambda: tf.saved_model.load(model_file_path),

                    # Approach 4: Load with explicit TensorFlow format
                    lambda: tf.keras.models.load_model(model_file_path,
                                                       compile=False,
                                                       options=tf.saved_model.LoadOptions(
                                                           experimental_io_device='/job:localhost'
                                                       )
                                                       )
                ]

                load_errors = []
                for i, load_attempt in enumerate(loading_approaches, 1):
                    try:
                        self.model = load_attempt()
                        print(f"Successfully loaded model using approach {i}")
                        break
                    except Exception as e:
                        load_errors.append(f"Approach {i} failed: {str(e)}")
                else:
                    raise ValueError(f"All loading approaches failed:\n" + "\n".join(load_errors))

                # Verify model structure and compatibility
                if isinstance(self.model, tf.keras.Model):
                    # Get input shape from model configuration
                    self.input_shape = tuple(self.model.input_shape[1:3])
                    output_shape = self.model.output_shape[-1]
                else:
                    # For SavedModel format, try to get shapes from signature
                    infer = self.model.signatures["serving_default"]
                    input_shape = infer.inputs[0].shape
                    output_shape = infer.outputs[0].shape[-1]
                    self.input_shape = tuple(input_shape[1:3])

                # Verify compatibility with labels
                if output_shape != len(self.class_labels):
                    raise ValueError(
                        f"Model output shape {output_shape} does not match number of class labels {len(self.class_labels)}")

                # Enable upload button and update UI
                self.upload_btn.setEnabled(True)
                self.result_label.setText(
                    f"Model loaded successfully!\nDetectable organs: {', '.join(self.class_labels)}\nYou can now upload medical organ images.")
                self.result_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px;
                        padding: 10px;
                        margin: 10px;
                        border-radius: 5px;
                        background-color: #e8f5e9;
                        color: #2e7d32;
                    }
                """)
                self.load_model_btn.setText("Change Model")

            except Exception as e:
                detailed_error = f"Error loading model: {str(e)}\n\nPlease ensure:"
                detailed_error += "\n- The model is compatible with TensorFlow 2.x"
                detailed_error += "\n- The model file is not corrupted"
                detailed_error += "\n- The model format matches the file extension"
                detailed_error += "\n- All required custom objects are properly defined"
                self.show_error(detailed_error)
                print(f"Detailed error: {str(e)}")

    def predict_with_model(self, image):
        """Wrapper for model prediction to handle different model types"""
        try:
            if isinstance(self.model, tf.keras.Model):
                return self.model.predict(image, verbose=0)
            elif isinstance(self.model, tf.Module):
                # Handle SavedModel format
                infer = self.model.signatures["serving_default"]
                input_tensor = tf.convert_to_tensor(image)
                result = infer(input_tensor)
                return list(result.values())[0].numpy()
            else:
                raise ValueError("Unsupported model type")
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")


    def upload_image(self):
        if not self.model or not self.class_labels:
            self.show_error("Please load a model and class labels first!")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Medical Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            try:
                # Load and preprocess image
                image = self.preprocess_image(file_path)

                # Make prediction using wrapper
                prediction = self.predict_with_model(np.expand_dims(image, axis=0))
                self.show_prediction(prediction[0], file_path)

            except Exception as e:
                self.show_error(f"Error processing image: {str(e)}")

    def preprocess_image(self, file_path):
        # Read and preprocess image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to match model's input shape
        img = cv2.resize(img, self.input_shape)

        # Display the image
        self.display_image(img)

        # Normalize pixel values
        img = img.astype('float32') / 255.0

        return img

    def display_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def show_prediction(self, prediction, file_path):
        # Get the highest confidence prediction
        max_confidence = np.max(prediction) * 100
        predicted_class = np.argmax(prediction)
        organ_name = self.class_labels[predicted_class]

        if max_confidence < self.CONFIDENCE_THRESHOLD:
            result_text = f"Low confidence detection ({max_confidence:.1f}%)\n"
            result_text += "Please provide a clearer image of the organ.\n\n"
            result_text += "Tips for better images:\n"
            result_text += "- Ensure good lighting\n"
            result_text += "- Center the organ in the image\n"
            result_text += "- Minimize blur and noise\n"
            result_text += "- Use proper medical imaging protocols"

            self.result_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    padding: 10px;
                    margin: 10px;
                    border-radius: 5px;
                    background-color: #fff3e0;
                    color: #e65100;
                }
            """)
        else:
            # Show all confidence scores
            result_text = f"Detected Organ: {organ_name}\n"
            result_text += f"Confidence: {max_confidence:.1f}%\n\n"
            result_text += "Detailed Analysis:\n"

            # Sort predictions by confidence
            sorted_indices = np.argsort(prediction)[::-1]
            for idx in sorted_indices:
                confidence = prediction[idx] * 100
                organ = self.class_labels[idx]
                result_text += f"{organ}: {confidence:.1f}%\n"

            self.result_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    padding: 10px;
                    margin: 10px;
                    border-radius: 5px;
                    background-color: #e3f2fd;
                    color: #1565c0;
                }
            """)

        self.result_label.setText(result_text)

    def show_error(self, message):
        self.result_label.setText(f"Error: {message}")
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 10px;
                margin: 10px;
                border-radius: 5px;
                background-color: #ffebee;
                color: #c62828;
            }
        """)


def main():
    app = QApplication(sys.argv)
    window = OrganClassifierWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()