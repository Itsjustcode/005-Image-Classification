import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights="imagenet")

def get_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for the given image array."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_occlusion(image_path, heatmap, method="black_box", alpha=0.6):
    """Apply occlusion to the highlighted area of an image based on the heatmap."""
    img = cv2.imread(image_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Threshold the heatmap to identify the highlighted area
    threshold = 0.5
    mask = heatmap_resized > threshold

    if method == "black_box":
        # Apply a black box to the highlighted area
        img[mask] = [0, 0, 0]
    elif method == "blur":
        # Apply a Gaussian blur to the highlighted area
        blurred_img = cv2.GaussianBlur(img, (15, 15), 0)
        img[mask] = blurred_img[mask]
    elif method == "noise":
        # Add random noise to the highlighted area
        noise = np.random.randint(0, 256, img.shape, dtype=np.uint8)
        img[mask] = noise[mask]

    return img

def classify_and_occlude(image_path):
    """Classify an image, apply Grad-CAM, and test occlusions."""
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the top-3 classes
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # Get the class index of the top prediction
        top_pred_index = np.argmax(predictions[0])

        # Generate Grad-CAM heatmap
        last_conv_layer_name = "Conv_1"  # Last convolutional layer in MobileNetV2
        heatmap = get_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=top_pred_index)

        # Apply occlusions and save results
        for method in ["black_box", "blur", "noise"]:
            occluded_img = apply_occlusion(image_path, heatmap, method=method)
            output_path = f"occluded_{method}.png"
            cv2.imwrite(output_path, occluded_img)
            print(f"Occlusion applied with '{method}' method and saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    # Path to the input image
    image_path = "Golden_Two-unsplash.jpg"
    classify_and_occlude(image_path)
