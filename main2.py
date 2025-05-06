from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from rembg import remove
from PIL import Image
import cv2
import numpy as np
import io
from gfpgan import GFPGANer
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Define input and output folders
INPUT_FOLDER = "images/input"
OUTPUT_FOLDER = "images/output"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize GFPGAN
gfpgan = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",  # Path to the GFPGAN model
    upscale=1,  # Upscale factor (1 for no upscaling)
    arch="clean",  # Architecture type
    channel_multiplier=2,
    bg_upsampler=None,
)

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

def enhance_image_with_gfpgan(image_path):
    """
    Enhance the image quality using GFPGAN.
    """
    print("[DEBUG] Enhancing image quality with GFPGAN...")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    try:
        # Use GFPGAN to enhance the image
        _, _, restored_img = gfpgan.enhance(
            img,
            has_aligned=False,  # Set to True if faces are already aligned
            only_center_face=False,  # Set to True to process only the center face
            paste_back=True,  # Paste the restored face back to the original image
        )
        return restored_img
    except Exception as e:
        print(f"[DEBUG] GFPGAN enhancement failed: {str(e)}")
        # Return the original image if enhancement fails
        return img
    
# Add this function after add_white_border function
def add_frame_overlay(image, frame_path):
    """
    Add a frame overlay to the image.
    """
    # Read the frame image with alpha channel
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    
    # Resize frame to match the image dimensions
    frame = cv2.resize(frame, (image.shape[1], image.shape[0]))
    
    # Extract alpha channel and normalize
    alpha = frame[:, :, 3] / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])
    
    # Composite the frame over the image
    result = image.copy()
    for c in range(0, 3):
        result[:, :, c] = (
            alpha[:, :, c] * frame[:, :, c] +
            (1 - alpha[:, :, c]) * image[:, :, c]
        )
    
    return result

# Modify the process_images function - add frame overlay before creating the buffer
def process_images(src_image_path, bgr_image_path, output_path):
    """
    Process images: enhance quality, remove background, and composite over the background image.
    """
    print(f"[DEBUG] Processing images: {src_image_path}, {bgr_image_path}")

    # Step 1: Resize the source image to increase its size
    print("[DEBUG] Resizing source image...")
    src_image = cv2.imread(src_image_path)
    scale_factor = 3.0  # Changed from 1.5 to 3.0 for larger initial size
    new_width = int(src_image.shape[1] * scale_factor)
    new_height = int(src_image.shape[0] * scale_factor)
    resized_src_image = cv2.resize(src_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_src_image_path = os.path.join(OUTPUT_FOLDER, "resized_src_image.png")
    cv2.imwrite(resized_src_image_path, resized_src_image)
    print(f"[DEBUG] Resized source image saved to: {resized_src_image_path}")

    # Step 2: Enhance the resized source image quality using GFPGAN
    enhanced_image = enhance_image_with_gfpgan(resized_src_image_path)
    enhanced_image_path = os.path.join(OUTPUT_FOLDER, "enhanced_src_image.png")
    cv2.imwrite(enhanced_image_path, enhanced_image)
    print(f"[DEBUG] Enhanced image saved to: {enhanced_image_path}")

  

    # Step 3: Remove background from the padded source image
    with open(enhanced_image_path, "rb") as input_file:
        input_image = input_file.read()

    print("[DEBUG] Background removal started...")
    output_image = remove(input_image)
    print("[DEBUG] Background removal completed.")

    # Save the image with background removed
    no_bg_image_path = os.path.join(OUTPUT_FOLDER, "no_bg.png")
    with open(no_bg_image_path, "wb") as output_file:
        output_file.write(output_image)
    print(f"[DEBUG] Background-removed image saved to: {no_bg_image_path}")

    # Step 4: Composite the result over the background image using OpenCV
    print("[DEBUG] Compositing images...")
    background = cv2.imread(bgr_image_path)
    foreground = cv2.imread(no_bg_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

    # Resize the foreground to fit within the background while maintaining aspect ratio
    bg_height, bg_width = background.shape[:2]
    fg_height, fg_width = foreground.shape[:2]
    scale = min(bg_width / fg_width, bg_height / fg_height) * 3.0  # Changed from 1.3 to 3.0
    new_fg_width = int(fg_width * scale)
    new_fg_height = int(fg_height * scale)
    resized_foreground = cv2.resize(foreground, (new_fg_width, new_fg_height), interpolation=cv2.INTER_AREA)

    # Crop the resized foreground to fit within the background dimensions
    if new_fg_width > bg_width:
        x_crop = (new_fg_width - bg_width) // 2
        resized_foreground = resized_foreground[:, x_crop:x_crop+bg_width]
        new_fg_width = bg_width
    if new_fg_height > bg_height:
        y_crop = (new_fg_height - bg_height) // 2
        resized_foreground = resized_foreground[y_crop:y_crop+bg_height, :]
        new_fg_height = bg_height

    # Calculate the position to center the resized foreground horizontally and place it at the bottom
    x_offset = (bg_width - new_fg_width) // 2
    y_offset = bg_height - new_fg_height

    # Extract the alpha channel from the resized foreground
    alpha = resized_foreground[:, :, 3] / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # Composite the images
    for c in range(0, 3):
        background[y_offset:y_offset+new_fg_height, x_offset:x_offset+new_fg_width, c] = (
            alpha[:, :, c] * resized_foreground[:, :, c] +
            (1 - alpha[:, :, c]) * background[y_offset:y_offset+new_fg_height, x_offset:x_offset+new_fg_width, c]
        )

    # After adding white border and before creating buffer
    frame_path = "frame.png"  # Make sure frame.png is in the same directory as main2.py
    if os.path.exists(frame_path):
        print("[DEBUG] Adding frame overlay...")
        background = add_frame_overlay(background, frame_path)
        print("[DEBUG] Frame overlay added.")

    # Save the final composited image
    cv2.imwrite(output_path, background)
    print(f"[DEBUG] Final composited image saved to: {output_path}")

    # Convert the composited image to a PNG buffer
    _, buffer = cv2.imencode(".png", background)
    png_buffer = io.BytesIO(buffer)
    print("[DEBUG] Composited image converted to PNG buffer.")

    return png_buffer


@app.route("/api/removebg/", methods=["POST"])
def upload_images():
    try:
        print("[DEBUG] Received request to /api/swap-face/")

        # Check if files are present in the request
        if "sourceImage" not in request.files or "bgr_image" not in request.files:
            print("[DEBUG] Missing source or background image in request.")
            return jsonify({"error": "Both source and background images are required."}), 400

        # Get the uploaded files
        src_image = request.files["bgr_image"]  # Changed from src_image to sourceImage
        bgr_image = request.files["sourceImage"]

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
    
    
    
def remove_background_with_clipdrop(image_path):
    """
    Remove background using ClipDrop API.
    """
    print("[DEBUG] Removing background using ClipDrop API...")
    api_key = "b54780508fd1d61abff1eb2eaa6eaa4b157ffb81e4328a4e7a428cb227cdd89053193f68473f838eb0466c2174195482"  # Replace with your ClipDrop API key
    url = "https://clipdrop-api.co/remove-background/v1"

    with open(image_path, "rb") as image_file:
        response = requests.post(
            url,
            files={"image_file": image_file},
            headers={"x-api-key": api_key}
        )

    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"ClipDrop API request failed with status code {response.status_code}: {response.text}")
    
    
    
@app.route("/api/clip-drop/", methods=["POST"])
def upload_images_clipdrop():
    try:
        print("[DEBUG] Received request to /api/clip-drop/")

        # Check if files are present in the request
        if "sourceImage" not in request.files or "bgr_image" not in request.files:
            print("[DEBUG] Missing source or background image in request.")
            return jsonify({"error": "Both source and background images are required."}), 400

        # Get the uploaded files
        src_image = request.files["bgr_image"]
        bgr_image = request.files["sourceImage"]

        # Save the uploaded files to the input folder
        src_image_path = os.path.join(INPUT_FOLDER, "src_image.jpg")
        bgr_image_path = os.path.join(INPUT_FOLDER, "bgr_image.jpg")
        src_image.save(src_image_path)
        bgr_image.save(bgr_image_path)
        print(f"[DEBUG] Source image saved to: {src_image_path}")
        print(f"[DEBUG] Background image saved to: {bgr_image_path}")

        # Define the output path
        output_path = os.path.join(OUTPUT_FOLDER, "final_output.png")
        clipdrop_output_path = os.path.join(OUTPUT_FOLDER, "clipdropg.png")

        # Process the images
        print("[DEBUG] Starting image processing with ClipDrop...")
        enhanced_image = enhance_image_with_gfpgan(src_image_path)
        enhanced_image_path = os.path.join(OUTPUT_FOLDER, "enhanced_src_image.png")
        cv2.imwrite(enhanced_image_path, enhanced_image)
        print(f"[DEBUG] Enhanced image saved to: {enhanced_image_path}")

        # Remove background using ClipDrop API
        print("[DEBUG] Background removal started with ClipDrop...")
        output_image = remove_background_with_clipdrop(enhanced_image_path)
        print("[DEBUG] Background removal completed with ClipDrop.")

        # Save the image with background removed
        with open(clipdrop_output_path, "wb") as output_file:
            output_file.write(output_image)
        print(f"[DEBUG] Background-removed image saved to: {clipdrop_output_path}")

        # Composite the result over the background image using OpenCV
        print("[DEBUG] Compositing images...")
        background = cv2.imread(bgr_image_path)
        foreground = cv2.imread(clipdrop_output_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

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

        # Return the processed image as a response
        png_buffer.seek(0)  # Reset buffer position to the beginning
        return send_file(png_buffer, mimetype="image/png")

    except Exception as e:
        print(f"[DEBUG] Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[DEBUG] Starting Flask server on port 3000...")
    app.run(host="0.0.0.0", port=3000, debug=False)  # Set debug=False