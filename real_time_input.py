import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont # For image processing and drawing text/shapes
import numpy as np
import cv2 # OpenCV for webcam access, face detection, and displaying video frames
import sys # Used to detect the operating system for font path selection

# --- 0. Configuration and Setup ---

# Set the device to run the model on (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directory where your trained model weights are saved
model_save_dir = "models"

# List of emotion classes. This order MUST match the order used during model training.
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(classes)

# Path to your best saved Vision Transformer model weights
best_pth_model_path = os.path.join(model_save_dir, "best_emotion_vit.pth")

# --- IMPORTANT: This value MUST match the `blocks_to_unfreeze` used during your ViT training ---
# It's crucial for correctly re-creating the model's architecture before loading its saved weights.
FINE_TUNED_VIT_BLOCKS = 4

# --- OpenCV DNN Face Detector Setup ---
# Paths to the pre-trained Caffe model files for face detection.
# These files are essential for finding faces in the webcam feed.
FACE_DETECTOR_PROTO = 'D:/Emotion_Detection/OpenCV_DNN_Face_Detector/deploy.prototxt.txt'
FACE_DETECTOR_MODEL = 'D:/Emotion_Detection/OpenCV_DNN_Face_Detector/res10_300x300_ssd_iter_140000.caffemodel'

face_detector_net = None # Initialize face detector object to None
if os.path.exists(FACE_DETECTOR_PROTO) and os.path.exists(FACE_DETECTOR_MODEL):
    try:
        # Load the pre-trained Caffe model for fast and accurate face detection
        face_detector_net = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_PROTO, FACE_DETECTOR_MODEL)
        print(f"OpenCV DNN face detector loaded.")
    except Exception as e:
        print(f"Error loading OpenCV DNN face detector: {e}. Face detection will be skipped.")
        face_detector_net = None # Ensure it's None if loading fails
else:
    print(f"OpenCV DNN face detector files not found. Face detection will be skipped.")
    print(f"   Ensure '{FACE_DETECTOR_PROTO}' and '{FACE_DETECTOR_MODEL}' are correct paths.")
    face_detector_net = None # Ensure it's None if files are missing

# --- 1. Data Transformations for Emotion Inference ---
# These transformations prepare each cropped face image for the emotion recognition model.
# They MUST be IDENTICAL to the 'test_transform' used during model training for consistent results.

# Standard ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

emotion_inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert to 3-channel grayscale (as trained)
    transforms.Resize((224, 224)),               # Resize to ViT's required input size
    transforms.ToTensor(),                       # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean, std)              # Normalize pixel values to standard range
])

# --- 2. Helper Function to Create ViT Model Architecture ---
# This function rebuilds the Vision Transformer's structure exactly as it was during training.
# This architecture is essential to load the saved model weights (.pth file) correctly.
def create_fine_tuned_vit_architecture(num_classes, device, blocks_to_unfreeze):
    from torchvision.models import ViT_B_16_Weights # Import specific ViT weights enumeration

    # Create the base Vision Transformer model structure without loading pre-trained ImageNet weights.
    # We use 'weights=None' because we're loading our own fine-tuned weights later.
    model = models.vit_b_16(weights=None)
    
    # Replace the final classification head to output our specific number of emotion classes.
    # The new linear layer will automatically be trainable (`requires_grad=True`).
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    # Freeze all model parameters initially. This is a common practice before selectively unfreezing.
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the parameters of the newly added (or adapted) classification head.
    # These parameters are directly involved in the final emotion classification.
    for param in model.heads.parameters():
        param.requires_grad = True

    # Fine-tune: Unfreeze a specified number of the last encoder blocks of the ViT.
    # These blocks will have their weights updated during fine-tuning to better adapt to your emotion data.
    num_encoder_blocks = len(model.encoder.layers)
    for i, block in enumerate(model.encoder.layers):
        if i >= (num_encoder_blocks - blocks_to_unfreeze): # Unfreeze from this block index onwards
            for param in block.parameters():
                param.requires_grad = True

    return model.to(device) # Move the configured model to the selected device (GPU/CPU)


# --- 3. Emotion Model Loading ---
# Load the trained emotion recognition model from its saved .pth file.
print("\n--- Loading Emotion Model ---")
loaded_emotion_model = None # Initialize model object to None
if os.path.exists(best_pth_model_path):
    try:
        # Create the model architecture exactly matching the training setup
        loaded_emotion_model = create_fine_tuned_vit_architecture(num_classes, device, FINE_TUNED_VIT_BLOCKS)
        # Load the saved trained weights into the created model architecture
        loaded_emotion_model.load_state_dict(torch.load(best_pth_model_path, map_location=device, weights_only=True))
        loaded_emotion_model.eval() # Set model to evaluation mode (crucial for consistent inference results)
        print(f"Emotion model loaded successfully from: {best_pth_model_path}")
    except Exception as e:
        print(f"Error loading emotion model: {e}")
else:
    print(f"Emotion model not found at: {best_pth_model_path}")


# --- 4. Real-time Emotion Detection Function ---
# This function captures video from the webcam, detects faces,
# predicts emotions for each face, and displays the results in real-time.
def real_time_emotion_detection(model, classes, transform, device, face_detector):
    # Ensure both emotion model and face detector are loaded before starting detection.
    if model is None or face_detector is None:
        print("Required models not loaded. Cannot perform real-time detection.")
        return

    # Open the default webcam (usually index 0). Change to 1, 2, etc. for other cameras.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check connection or if it's in use by another application.")
        return

    print("\n--- Starting Real-time Emotion Detection (Press 'q' to quit) ---")

    # Define colors for drawing bounding boxes and text on the video frame
    rect_color = (0, 255, 0)     # Green for bounding boxes
    text_color = (255, 255, 255) # White for text
    outline_color = (0, 0, 0)    # Black for text outline (improves readability on varying backgrounds)

    # --- PIL Font Setup for drawing high-quality text on frames ---
    # Attempt to load a high-quality TrueType font. If not found, it will fall back to a default font.
    pil_font_size = 18 # Base font size for emotion labels
    
    # Map common OS platforms to their typical font paths
    font_paths = {
        'win32': "C:/Windows/Fonts/arial.ttf", # Windows
    }
    # Get the correct font path for the current OS. 'sys.platform' returns values like 'win32', 'darwin', 'linux'.
    selected_font_path = font_paths.get(sys.platform, "arial.ttf") # Fallback to a common font name if OS not explicitly mapped
    
    # Load the TrueType font. If it fails (e.g., font file not found), use PIL's default font.
    current_pil_font = None
    try:
        current_pil_font = ImageFont.truetype(selected_font_path, pil_font_size)
    except IOError:
        print(f"  Warning: TrueType font not found at '{selected_font_path}'. Using default PIL font (may be lower quality).")
        current_pil_font = ImageFont.load_default()

    # Main loop for real-time video processing
    while True:
        ret, frame = cap.read() # Read a single frame from the webcam
        if not ret: # Check if the frame was read successfully
            print("Failed to grab frame, exiting...")
            break

        # Get frame dimensions for scaling face detection results
        (h, w) = frame.shape[:2]

        # Convert OpenCV's BGR frame to PIL's RGB format for drawing and model input.
        # PIL (Pillow) is used for high-quality text rendering and image manipulation.
        img_pil_overlay = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # --- Face Detection on the current frame ---
        face_locations = []
        # Prepare the frame for the DNN face detector: resize and normalize pixel values.
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward() # Run the face detection model

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7: # Filter out weak detections; 0.7 is a common threshold for good results
                # Scale bounding box coordinates back to the original frame size
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding box coordinates stay within the frame boundaries
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Optional: Add padding around the detected face for more context when cropping for emotion model.
                # This helps ensure the whole face is included even if detection is slightly off.
                padding_x = int((endX - startX) * 0.1)
                padding_y = int((endY - startY) * 0.1)
                startX = max(0, startX - padding_x)
                startY = max(0, startY - padding_y)
                endX = min(w, endX + padding_x)
                endY = min(h, endY + padding_y)

                # Store face location in (top, right, bottom, left) format for consistency with PIL/image processing.
                face_locations.append((startY, endX, endY, startX))

        # --- Process Each Detected Face and Draw Results ---
        # Create a drawing object for the current PIL image, allowing drawing operations.
        draw = ImageDraw.Draw(img_pil_overlay) 

        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw a green bounding box around the detected face on the PIL image.
            draw.rectangle([(left, top), (right, bottom)], outline=rect_color, width=3) 

            # Crop the face region from the current frame and prepare it for the emotion model.
            # This involves resizing and normalizing according to the 'emotion_inference_transform'.
            face_crop_pil = img_pil_overlay.crop((left, top, right, bottom))
            face_tensor = transform(face_crop_pil).unsqueeze(0).to(device) # Add batch dimension

            # Perform emotion prediction using the loaded model.
            with torch.no_grad(): # Disable gradient calculation for faster inference (no training involved here)
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1) # Convert raw outputs to probabilities
                predicted_label = classes[torch.argmax(probabilities).item()] # Get the most likely emotion label
                predicted_confidence = torch.max(probabilities).item() * 100 # Get its confidence score in percentage

            # --- Draw Emotion Text and Confidence on the Frame ---
            text_to_display = f"{predicted_label} ({predicted_confidence:.1f}%)"
            
            # Dynamically adjust font size based on the height of the detected face for better readability.
            # Ensures text is neither too small nor too large.
            face_height = bottom - top
            dynamic_font_size = max(10, min(int(face_height * 0.2), 30)) # Clamped between 10 (min) and 30 (max)
            
            # Re-load the font with the dynamic size. This ensures sharp text for each unique face size.
            # If the TrueType font couldn't be loaded initially, it falls back to PIL's default.
            try:
                current_pil_font_dynamic = ImageFont.truetype(selected_font_path, dynamic_font_size)
            except IOError:
                current_pil_font_dynamic = ImageFont.load_default()

            # Calculate text dimensions to determine the size of the background rectangle for the text.
            text_bbox = draw.textbbox((0, 0), text_to_display, font=current_pil_font_dynamic)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_padding_x = 5 # Padding around text inside its background box
            text_padding_y = 5
            
            # Determine text block position: prefer to place text above the bounding box.
            # If placing above goes off-screen, place it below the bounding box instead.
            text_x = left
            text_y = top - text_height - text_padding_y 
            if text_y < 0: # Check if text goes off top of the image
                text_y = bottom + text_padding_y # Place below the box

            # Ensure text box stays within horizontal frame boundaries.
            if text_x + text_width + 2*text_padding_x > w: # 'w' is the frame width
                text_x = w - (text_width + 2*text_padding_x) # Move text left if it's too far right
                text_x = max(0, text_x) # Ensure it doesn't go off the left edge

            # Draw a semi-transparent black background rectangle for the text.
            # This significantly improves text readability on complex backgrounds.
            overlay_alpha = 150 # Transparency level (0-255, 255 is opaque)
            rect_img = Image.new('RGBA', (text_width + 2*text_padding_x, text_height + 2*text_padding_y), (0, 0, 0, overlay_alpha))
            img_pil_overlay.paste(rect_img, (text_x, text_y), rect_img) # Paste this rectangle onto the main image
            
            # Re-initialize the drawing object after pasting the overlay.
            # This is necessary so subsequent drawing operations are applied on the updated image with the background.
            draw = ImageDraw.Draw(img_pil_overlay)

            # Draw text with a black outline first, then the main white text.
            # This technique makes text stand out clearly against any background color.
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]: # Iterates 4 directions for outline
                draw.text((text_x + text_padding_x + dx, text_y + text_padding_y + dy), text_to_display, font=current_pil_font_dynamic, fill=outline_color)
            # Draw the main white text on top of the black outline
            draw.text((text_x + text_padding_x, text_y + text_padding_y), text_to_display, font=current_pil_font_dynamic, fill=text_color)
            
        # Convert the modified PIL image (with drawings) back to OpenCV's BGR format.
        # OpenCV's `imshow` function expects BGR images.
        img_cv2_display = cv2.cvtColor(np.array(img_pil_overlay), cv2.COLOR_RGB2BGR)

        # Display the video frame in a window titled 'Real-time Emotion Detection'.
        cv2.imshow('Real-time Emotion Detection', img_cv2_display)

        # Wait for 1 millisecond for a key press. If 'q' is pressed, break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam resource and close all OpenCV display windows when the loop ends.
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Real-time Detection Stopped ---")

# --- 5. Main Execution Block ---
# This code runs only when the script is executed directly (not imported as a module).
if __name__ == "__main__":
    # Start the real-time emotion detection if both the emotion model and face detector were loaded successfully.
    if loaded_emotion_model and face_detector_net:
        real_time_emotion_detection(loaded_emotion_model, classes, emotion_inference_transform, device, face_detector_net)
    else:
        print("\nReal-time detection cannot start: Emotion model or Face Detector not loaded properly. Please check console for errors.")