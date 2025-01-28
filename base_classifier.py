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
    # Get the gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradients of the top predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Get gradients of the predicted class with respect to the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Compute the guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map by its corresponding gradient
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(heatmap, img_path, alpha=0.6):
    """Overlay the heatmap on the original image."""
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Combine heatmap with the original image
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return overlay

def classify_and_generate_grad_cam(image_path):
    """Classify an image, generate Grad-CAM heatmap, and display results."""
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

        # Overlay heatmap on the original image
        overlay_image = overlay_heatmap_on_image(heatmap, image_path)

        # Save and display the heatmap
        heatmap_output_path = "grad_cam_overlay.png"
        cv2.imwrite(heatmap_output_path, overlay_image)
        print(f"Grad-CAM heatmap saved as '{heatmap_output_path}'.")
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    # Path to the input image
    image_path = "Golden_Two-unsplash.jpg"
    classify_and_generate_grad_cam(image_path)
