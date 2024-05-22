import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = load_model('src/model.keras')



def generate_tags(file_path):
    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)
    with open("src/categories.txt", "r") as f:
        lines = f.readlines()

    class_labels = []
    for line in lines:
        line = line.strip()  # Remove whitespace
        class_labels.extend(line.split(","))

    class_labels = [label.strip('"') for label in class_labels]

    print("Loaded class labels:", class_labels)

    # Select top 5 predictions with probabilities, handling potential errors:
    try:
        top_5_predictions = sorted(zip(class_labels, prediction[0]), key=lambda item: item[1], reverse=True)[:5]
    except IndexError:  # Handle cases where the number of labels may not match the prediction length
        print("Warning: Index error encountered.")
        top_5_predictions = []
    top_5_labels = [item[0] for item in top_5_predictions]
    return top_5_labels


def preprocess_image(image_path):
    """Preprocesses an image for prediction."""
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0