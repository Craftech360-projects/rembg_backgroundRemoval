import os
import cv2

from main2 import OUTPUT_FOLDER, enhance_image_with_gfpgan


def process_images(src_image_path, bgr_image_path, output_path):
    """
    Process images: enhance quality, remove background, and composite over the background image.
    """
    print(f"[DEBUG] Processing images: {src_image_path}, {bgr_image_path}")

    # Step 1: Enhance the source image quality using GFPGAN
    enhanced_image = enhance_image_with_gfpgan(src_image_path)
    enhanced_image_path = os.path.join(OUTPUT_FOLDER, "enhanced_src_image.png")
    cv2.imwrite(enhanced_image_path, enhanced_image)
    print(f"[DEBUG] Enhanced image saved to: {enhanced_image_path}")

    # Add padding to the enhanced image to increase its width
    padding_size = 50  # Adjust the padding size as needed
    padded_image = cv2.copyMakeBorder(
        enhanced_image,
        top=0,
        bottom=0,
        left=padding_size,
        right=padding_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # White color in BGR format
    )
    padded_image_path = os.path.join(OUTPUT_FOLDER, "padded_src_image.png")
    cv2.imwrite(padded_image_path, padded_image)
    print(f"[DEBUG] Padded image saved to: {padded_image_path}")

    # Step 2: Remove background from the padded source image
    with open(padded_image_path, "rb") as input_file:
        input_image = input_file.read()

    print("[DEBUG] Background removal started...")
    output_image = os.remove(input_image)
    print("[DEBUG] Background removal completed.")

    # Save the image with background removed
    no_bg_image_path = os.path.join(OUTPUT_FOLDER, "no_bg.png")
    with open(no_bg_image_path, "wb") as output_file:
        output_file.write(output_image)
    print(f"[DEBUG] Background-removed image saved to: {no_bg_image_path}")

    # Step 3: Composite the result over the background image using OpenCV
    print("[DEBUG] Compositing images...")
    background = cv2.imread(bgr_image_path)
    foreground = cv2.imread(no_bg_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

    # Resize the foreground to fit within the background while maintaining aspect ratio
    bg_height, bg_width = background.shape[:2]
    fg_height, fg_width = foreground.shape[:2]
    scale = min(bg_width / fg_width, bg_height / fg_height) * 1.3  # Scale up by a factor of 1.3
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

    # Save the final composited image
    cv2.imwrite(output_path, background)
    print(f"[DEBUG] Final composited image saved to: {output_path}")

    # Convert the composited image to a PNG buffer
    _, buffer = cv2.imencode(".png", background)
    png_buffer = io.BytesIO(buffer)
    print("[DEBUG] Composited image converted to PNG buffer.")

    return png_buffer

# Example usage
src_image_path = "images/input/bgr_image.jpg"
bgr_image_path = "images/output/no_bg.png"
output_path = "images/output/final_output.png"
process_images(src_image_path, bgr_image_path, output_path)