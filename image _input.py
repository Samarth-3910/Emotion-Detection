import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont # For image handling and drawing on images
import matplotlib.pyplot as plt
import numpy as np
import cv2 # OpenCV for image manipulation and face detection

# --- 0. Configuration and Setup ---

# Set device for running the model (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directory where your trained model weights are saved
model_save_dir = "models"

# List of emotion classes. This MUST match the order used during training.
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(classes)

# Path to your best saved model weights for the Vision Transformer
best_pth_model_path = os.path.join(model_save_dir, "best_emotion_vit.pth")

# --- IMPORTANT: Set this to the number of encoder blocks you unfroze during ViT training --- (e.g., 4, 5, or 6)
# For example, if you unfroze the last 4 blocks of the Vision Transformer's encoder layers.()
FINE_TUNED_VIT_BLOCKS = 4

# --- OpenCV DNN Face Detector Setup ---
# Paths to the pre-trained Caffe model files for face detection.
# Ensure these files are correctly located relative to your script.
FACE_DETECTOR_PROTO = 'D:/Emotion_Detection/OpenCV_DNN_Face_Detector/deploy.prototxt.txt'
FACE_DETECTOR_MODEL = 'D:/Emotion_Detection/OpenCV_DNN_Face_Detector/res10_300x300_ssd_iter_140000.caffemodel'

face_detector_net = None
if os.path.exists(FACE_DETECTOR_PROTO) and os.path.exists(FACE_DETECTOR_MODEL):
    try:
        # Load the pre-trained face detection model
        face_detector_net = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_PROTO, FACE_DETECTOR_MODEL)
        print(f"OpenCV DNN face detector loaded.")
    except Exception as e:
        print(f"Error loading OpenCV DNN face detector: {e}. Face detection will be skipped.")
        face_detector_net = None
else:
    print(f"OpenCV DNN face detector files not found. Face detection will be skipped.")
    face_detector_net = None

# --- 1. Data Transformations for Emotion Inference ---
# These transformations MUST be IDENTICAL to the 'test_transform' used during model training.
# They prepare a cropped face image to be fed into the emotion recognition model.

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

emotion_inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert to 3 channels (as trained)
    transforms.Resize((224, 224)),             # Resize to ViT input size
    transforms.ToTensor(),                     # Convert to PyTorch Tensor
    transforms.Normalize(mean, std)            # Normalize pixel values
])

# --- 2. Helper Function to Create ViT Model Architecture ---
# This function re-creates the exact Vision Transformer architecture that was trained.
# It's needed to load the saved model weights (.pth file).
def create_fine_tuned_vit_architecture(num_classes, device, blocks_to_unfreeze):
    from torchvision.models import ViT_B_16_Weights # Import specific ViT weights enum

    # Create the Vision Transformer Base model (without pre-trained weights initially)
    model = models.vit_b_16(weights=None)
    
    # Replace the final classification head to match our number of emotion classes
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the new classifier head (it's already trainable, but good for clarity)
    for param in model.heads.parameters():
        param.requires_grad = True

    # Unfreeze the last few encoder blocks for fine-tuning, matching training setup
    num_encoder_blocks = len(model.encoder.layers)
    for i, block in enumerate(model.encoder.layers):
        if i >= (num_encoder_blocks - blocks_to_unfreeze):
            for param in block.parameters():
                param.requires_grad = True

    return model.to(device) # Move model to GPU/CPU

# --- 3. Emotion Model Loading ---
# Load the pre-trained emotion recognition model from the saved .pth file.
print("\n--- Loading Emotion Model ---")
loaded_emotion_model = None
if os.path.exists(best_pth_model_path):
    try:
        # Create the model architecture
        loaded_emotion_model = create_fine_tuned_vit_architecture(num_classes, device, FINE_TUNED_VIT_BLOCKS)
        # Load the saved weights into the model
        loaded_emotion_model.load_state_dict(torch.load(best_pth_model_path, map_location=device, weights_only=True))
        loaded_emotion_model.eval() # Set model to evaluation mode (important for inference)
        print(f"Emotion model loaded successfully from: {best_pth_model_path}")
    except Exception as e:
        print(f"Error loading emotion model: {e}")
else:
    print(f"Emotion model not found at: {best_pth_model_path}")

# --- 4. Prediction and Display Function ---
# This function detects faces, predicts emotions, and displays results on an image.
def predict_and_display_beautifully(model, image_path, classes, transform, device, face_detector, top_n=3):
    """
    Loads an image, detects faces, performs emotion prediction for each face,
    and displays primary prediction overlaid on the image.
    """
    if model is None:
        print(f"Skipping prediction for {os.path.basename(image_path)}: Emotion model not loaded.")
        return

    # Try to infer true label from image path (useful for test images)
    true_label_overall_image = "N/A"
    path_parts = image_path.split(os.sep)
    if len(path_parts) >= 3 and (path_parts[-3] == 'train' or path_parts[-3] == 'test'):
        true_label_overall_image = path_parts[-2]

    try:
        # Load image using PIL for display and OpenCV for face detection
        img_pil_original = Image.open(image_path).convert('RGB')
        img_cv2_original = np.array(img_pil_original) # Convert to NumPy array (RGB)
        img_cv2_bgr = cv2.cvtColor(img_cv2_original, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV DNN

        (h, w) = img_cv2_bgr.shape[:2]
        rect_color = (0, 255, 0) # Green color for bounding box
        text_color = (255, 255, 255) # White color for text
        outline_color = (0, 0, 0) # Black outline for text readability

        # Load a common TrueType font for better quality text on images
        pil_font = None 
        pil_font_size = 15 # Main text size

        # --- Try specifying a full path to a common font on your OS ---
        # You can usually find fonts in C:\Windows\Fonts

        font_file_path = "C:/Windows/Fonts/arial.ttf"

        try:
            pil_font = ImageFont.truetype(font_file_path, pil_font_size)
        except IOError:
            print(f"  Warning:Font not found at '{font_file_path}', falling back to default PIL font (may be lower quality).")
            pil_font = ImageFont.load_default()

        # --- Face Detection ---
        face_locations = []
        if face_detector:
            # Prepare image for DNN: resize, normalize pixel values
            blob = cv2.dnn.blobFromImage(cv2.resize(img_cv2_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            detections = face_detector.forward() # Run face detection

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7: # Only consider detections with high confidence
                    # Scale bounding box coordinates back to original image size
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure coordinates are within image boundaries
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # Store face location in (top, right, bottom, left) format
                    face_locations.append((startY, endX, endY, startX))

        # --- Process each detected face & Draw on Image ---
        img_pil_overlay = img_pil_original.copy() # Make a copy to draw on

        if len(face_locations) > 0:
            print(f"  Detected {len(face_locations)} face(s). Predicting emotions...")
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                draw = ImageDraw.Draw(img_pil_overlay) # Get drawing object
                
                # Draw green bounding box around the face
                draw.rectangle([(left, top), (right, bottom)], outline=rect_color, width=3)

                # Crop the face, apply transformations, and prepare for model
                face_crop_pil = img_pil_original.crop((left, top, right, bottom))
                face_tensor = transform(face_crop_pil).unsqueeze(0).to(device)

                # Predict emotion for this cropped face
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    top_probs, top_indices = torch.topk(probabilities, top_n)
                    predicted_label = classes[top_indices.cpu().numpy().flatten()[0]]
                    predicted_confidence = top_probs.cpu().numpy().flatten()[0] * 100

                # --- Prepare and Draw Text for this Face ---
                face_text = f"{predicted_label} ({predicted_confidence:.1f}%)"
                
                # Calculate text size for background rectangle
                text_bbox = draw.textbbox((0, 0), face_text, font=pil_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_padding_x = 5
                text_padding_y = 5
                
                # Position text above the bounding box, or below if it goes off the top
                text_start_x = left
                text_start_y = top - text_height - text_padding_y
                if text_start_y < 0:
                    text_start_y = bottom + text_padding_y

                # Adjust text position to stay within image bounds horizontally
                if text_start_x + text_width + 2*text_padding_x > img_pil_overlay.width:
                    text_start_x = img_pil_overlay.width - text_width - 2*text_padding_x
                    text_start_x = max(0, text_start_x)

                # Draw a semi-transparent black background for the text
                overlay = Image.new('RGBA', img_pil_overlay.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [(text_start_x, text_start_y), 
                     (text_start_x + text_width + 2*text_padding_x, text_start_y + text_height + 2*text_padding_y)],
                    fill=(0, 0, 0, 150) # Black with 150/255 transparency
                )
                img_pil_overlay = Image.alpha_composite(img_pil_overlay.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img_pil_overlay) # Re-initialize draw object after compositing

                # Draw text with a black outline for better visibility
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text((text_start_x + text_padding_x + dx, text_start_y + text_padding_y + dy), face_text, font=pil_font, fill=outline_color)
                # Draw the main white text
                draw.text((text_start_x + text_padding_x, text_start_y + text_padding_y), face_text, font=pil_font, fill=text_color)
                
                # --- Print Detailed Terminal Output for this Face ---
                print(f"\n  ═════ Face {i+1} (at [{left},{top}] {right-left}x{bottom-top}) ═════")
                print(f"  Predicted Emotion: \033[1m{predicted_label}\033[0m (Confidence: {predicted_confidence:.2f}%)")
                print(f"  Top {top_n} Probabilities:")
                for j in range(top_n):
                    label = classes[top_indices.cpu().numpy().flatten()[j]]
                    prob = top_probs.cpu().numpy().flatten()[j] * 100
                    print(f"    - {label}: {prob:.2f}%")
                print(f"  ═════════════════════════════")

        else:
            print("  No faces detected in this image. Emotion prediction will not be performed.")
            
        # --- Prepare Plot Title ---
        plot_title_lines = [f"File: {os.path.basename(image_path)}"]
        if true_label_overall_image != "N/A":
            plot_title_lines.append(f"True Label: {true_label_overall_image}")
        if len(face_locations) == 0:
            plot_title_lines.append("No Faces Detected for Emotion Prediction")
        else:
            plot_title_lines.append(f"Detected {len(face_locations)} Face(s)")
        
        full_plot_title = "\n".join(plot_title_lines)

        # --- Display the Resulting Image ---
        plt.figure(figsize=(10, 10)) # Adjust figure size
        plt.imshow(img_pil_overlay) # Display the PIL image with drawings
        plt.axis('off') # Hide axes
        plt.title(full_plot_title, fontsize=12, fontweight='bold', ha='center', va='center')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction for {os.path.basename(image_path)}: {e}")

# --- 5. Main Execution Block ---
# This part runs when the script is executed directly.
if __name__ == "__main__":
    # --- IMPORTANT: Set the path to the image you want to predict on here ---
    image_to_predict = 'D:/Emotion_Detection/image3.png' # Example path

    # Run the prediction and display function
    predict_and_display_beautifully(loaded_emotion_model, image_to_predict, classes, emotion_inference_transform, device, face_detector_net, top_n=3)

    print("\n--- Prediction Process Complete ---")