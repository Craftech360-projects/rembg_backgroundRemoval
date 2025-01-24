from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from rembg import remove
from PIL import Image
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # Enable CORS

# Define input and output folders
INPUT_FOLDER = "images/input"
OUTPUT_FOLDER = "images/output"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def add_white_border(image, border_size=20):
    """
    Add a white border around the image.
    """
    return cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # White color in BGR format
    )

def process_images(src_image_path, bgr_image_path, output_path):
    """
    Process images: remove background and composite over the background image.
    """
    print(f"[DEBUG] Processing images: {src_image_path}, {bgr_image_path}")

    # Step 1: Remove background from the source image
    with open(src_image_path, "rb") as input_file:
        input_image = input_file.read()

    print("[DEBUG] Background removal started...")
    output_image = remove(input_image)
    print("[DEBUG] Background removal completed.")

    # Save the image with background removed
    no_bg_image_path = os.path.join(OUTPUT_FOLDER, "no_bg.png")
    with open(no_bg_image_path, "wb") as output_file:
        output_file.write(output_image)
    print(f"[DEBUG] Background-removed image saved to: {no_bg_image_path}")

    # Step 2: Composite the result over the background image using OpenCV
    print("[DEBUG] Compositing images...")
    background = cv2.imread(bgr_image_path)
    foreground = cv2.imread(no_bg_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

    # Resize foreground to match background dimensions (if needed)
    foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    # Extract the alpha channel from the foreground
    alpha = foreground[:, :, 3] / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # Composite the images
    composite = (foreground[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)

    # Add a white border to the final image
    print("[DEBUG] Adding white border to the final image...")
    composite_with_border = add_white_border(composite, border_size=20)
    print("[DEBUG] White border added.")

    # Save the final composited image with border
    cv2.imwrite(output_path, composite_with_border)
    print(f"[DEBUG] Final composited image with border saved to: {output_path}")

    # Convert the composited image to a PNG buffer
    _, buffer = cv2.imencode(".png", composite_with_border)
    png_buffer = io.BytesIO(buffer)
    print("[DEBUG] Composited image converted to PNG buffer.")

    return png_buffer

@app.route("/upload-images", methods=["POST"])
def upload_images():
    try:
        print("[DEBUG] Received request to /upload-images")

        # Check if files are present in the request
        if "src_image" not in request.files or "bgr_image" not in request.files:
            print("[DEBUG] Missing source or background image in request.")
            return jsonify({"error": "Both source and background images are required."}), 400

        # Get the uploaded files
        src_image = request.files["src_image"]
        bgr_image = request.files["bgr_image"]

        # Save the uploaded files to the input folder
        src_image_path = os.path.join(INPUT_FOLDER, "src_image.jpg")
        bgr_image_path = os.path.join(INPUT_FOLDER, "bgr_image.jpg")
        src_image.save(src_image_path)
        bgr_image.save(bgr_image_path)
        print(f"[DEBUG] Source image saved to: {src_image_path}")
        print(f"[DEBUG] Background image saved to: {bgr_image_path}")

        # Define the output path
        output_path = os.path.join(OUTPUT_FOLDER, "final_output.png")

        # Process the images
        print("[DEBUG] Starting image processing...")
        png_buffer = process_images(src_image_path, bgr_image_path, output_path)
        print("[DEBUG] Image processing completed.")

        # Return the processed image as a response
        png_buffer.seek(0)  # Reset buffer position to the beginning
        return send_file(png_buffer, mimetype="image/png")

    except Exception as e:
        print(f"[DEBUG] Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[DEBUG] Starting Flask server on port 3000...")
    app.run(host="0.0.0.0", port=3000, debug=True)

