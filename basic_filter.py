from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os

def apply_filters(image_path):
    """
    Apply Solarize, Posterize, Color Enhancement, and a Spaghetti-Like filter to an image
    and save the results.
    """
    try:
        # Check if the input file exists
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found. Make sure the image exists in the correct directory.")
            return

        # Load the image
        img = Image.open(image_path)
        print(f"Image loaded: {image_path}, Size: {img.size}, Mode: {img.mode}")

        # Ensure the image is in RGB format
        img = img.convert("RGB")
        print("Image converted to RGB format.")

        # Apply Solarize Effect
        try:
            solarized_img = ImageOps.solarize(img)
            solarized_img.save("solarized_result.png")
            print("Solarize effect applied and saved as 'solarized_result.png'.")
        except Exception as e:
            print(f"Error applying Solarize filter: {e}")

        # Apply Posterize Effect
        try:
            posterized_img = ImageOps.posterize(img, bits=3)  # Reduce to 3 bits per channel
            posterized_img.save("posterized_result.png")
            print("Posterize effect applied and saved as 'posterized_result.png'.")
        except Exception as e:
            print(f"Error applying Posterize filter: {e}")

        # Apply Color Enhancement
        try:
            enhancer = ImageEnhance.Color(img)
            enhanced_img = enhancer.enhance(2.0)  # Double the color saturation
            enhanced_img.save("color_enhanced_result.png")
            print("Color enhancement applied and saved as 'color_enhanced_result.png'.")
        except Exception as e:
            print(f"Error applying Color Enhancement filter: {e}")

        # Apply Spaghetti-Like Filter (Contour + Exaggerated Colors)
        try:
            print("Applying Spaghetti-like filter...")
            spaghetti_img = img.filter(ImageFilter.CONTOUR)  # Add contour effect
            enhancer = ImageEnhance.Color(spaghetti_img)
            spaghetti_img = enhancer.enhance(3.0)  # Triple the color saturation
            spaghetti_img.save("spaghetti_result.png")
            print("Spaghetti-like filter applied and saved as 'spaghetti_result.png'.")
        except Exception as e:
            print(f"Error applying Spaghetti-like filter: {e}")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    # Path to the input image
    image_path = "Golden_Two-unsplash.jpg"  # Replace with the correct image path
    apply_filters(image_path)
