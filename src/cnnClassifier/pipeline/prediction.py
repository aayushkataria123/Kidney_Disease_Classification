import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Load the model once
        self.model = load_model(os.path.join("model", "model.h5"))

        # Class label mapping (based on alphabetical directory order used in training)
        self.class_labels = {
            0: 'Cyst',
            1: 'Normal',
            2: 'Stone',
            3: 'Tumor'
        }

    def predict(self):
        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Ensure consistency with training

        # Predict
        prediction_array = self.model.predict(test_image)
        predicted_index = np.argmax(prediction_array, axis=1)[0]
        confidence = float(np.max(prediction_array))

        # Get class label
        prediction_label = self.class_labels.get(predicted_index, "Unknown")

        # Log the result (optional)
        print(f"Predicted class index: {predicted_index}, label: {prediction_label}, confidence: {confidence:.2f}")

        # Return result
        return [{
            "image": prediction_label,
            "confidence": f"{confidence:.2f}"
        }]
