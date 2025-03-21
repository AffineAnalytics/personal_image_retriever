import os
import faiss
import torch
from PIL import Image
import numpy
from transformers import AutoProcessor, AutoModel
from qai_hub_models.models.openai_clip.model import Clip
from qai_hub_models.models.mediapipe_face.model import MediaPipeFace
from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from torchvision import transforms
import pickle

def load_images(folder):
    """
    Loads image file paths from a given folder.
    Returns a list of image paths for supported formats (.jpg, .jpeg, .png, .JPG).
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):  # Filter for supported image formats
            images.append(filename)
    return images

def load_model(device, model_choice):
    """
    Loads a specified multimodal model onto the given device (CPU/GPU).
    Returns the processor and model.
    """
    if model_choice == "CLIP":
        processor = None  # CLIP doesn't need a separate processor
        model = Clip.from_pretrained()  # Load the pre-trained CLIP model
    # Uncomment below for JINA model support
    # elif model_choice == "JINA":
    #     processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
    #     model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
    else:
        print("Please select a valid model")
        return None
    return processor, model

def transform_image(device, image):
    """
    Applies transformations to an image: resizing, converting to tensor, and normalizing.
    Returns a processed image tensor.
    """
    image = image.convert('RGB')  # Ensure image is in RGB format
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize with CLIP values
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

def get_image_embedding(image_path, processor, model, device, model_choice):
    """
    Extracts an image embedding using the specified processor and model.
    Returns a normalized embedding tensor.
    """
    image = Image.open(image_path)  # Open the image
    if model_choice == "CLIP":
        image = transform_image(device, image)  # Apply transformations
        with torch.no_grad():
            outputs = model.image_encoder.to(device)(image)  # Generate image embeddings
    else:
        # Alternative model handling (JINA or others)
        image = image.convert("RGB").resize((224, 224))  # Resize and convert to RGB
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)  # Extract image features
    return torch.nn.functional.normalize(outputs, p=2, dim=1)  # L2 normalize the embeddings

def get_text_embedding(text_query, processor, model, device, model_choice):
    """
    Extracts a text embedding using the specified processor and model.
    Returns a normalized embedding tensor.
    """
    if model_choice == "CLIP":
        inputs = model.tokenizer_func(text_query).to(device)  # Tokenize text
        with torch.no_grad():
            outputs = model.text_encoder.to(device)(inputs)  # Generate text embeddings
    else:
        # Alternative model handling (JINA or others)
        inputs = processor(text=text_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)  # Extract text features
    return torch.nn.functional.normalize(outputs, p=2, dim=1)  # L2 normalize the embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from a set of embeddings.
    Returns a FAISS index for similarity search.
    """
    embeddings_np = embeddings.numpy().astype('float32')  # Convert to NumPy array with float32 precision
    dimension = embeddings_np.shape[1]  # Get the dimension of embeddings
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner product search (for cosine similarity)
    faiss_index.add(embeddings_np)  # Add embeddings to the FAISS index
    return faiss_index

def load_fr_model(device):
    """
    Loads a face recognition model:
    - MediaPipe for face detection
    - CLIP for face feature extraction
    Returns the MediaPipe face model and CLIP image encoder.
    """
    model = MediaPipeFace.from_pretrained()  # Load the MediaPipe face detection model
    mediapipe_app = MediaPipeFaceApp(model=model)  # Initialize MediaPipe application
    clip_encoder = Clip.from_pretrained().image_encoder.to(device)  # Load CLIP image encoder

    return mediapipe_app, clip_encoder

def get_face_embeddings(img_path, mediapipe_app, clip_encoder, device):
    """
    Extracts face embeddings from an image using:
    - MediaPipe for face detection
    - CLIP for feature extraction
    Returns normalized face embeddings or None if no faces are detected.
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Ensure image is in RGB format
    embeddings = []

    # Face detection
    batched_selected_boxes, _, _, _ = mediapipe_app.predict_landmarks_from_image(img, raw_output=True)

    if batched_selected_boxes[0] is None:
        return None  # No faces detected
    else:
        for box in batched_selected_boxes[0]:
            # Extract face bounding box coordinates
            x1, y1 = box[0][0].int().item(), box[0][1].int().item()
            x2, y2 = box[1][0].int().item(), box[1][1].int().item()

            # Crop the detected face
            cropped_image = img.crop((x1, y1, x2, y2))
            cropped_image = transform_image(device, cropped_image)  # Apply transformations

            # Extract face features using CLIP
            emb = clip_encoder(cropped_image)
            embeddings.append(emb)
    return embeddings

def compute_rrf(rank_clip, rank_face, k=60):
    """
    Computes Reciprocal Rank Fusion (RRF) score for two ranked lists.
    Returns an RRF score for fusion-based ranking.
    """
    return 1 / (k + rank_clip) + 2 / (k + rank_face)

def normalize_embeddings(embeddings):
    """
    Normalizes embeddings using L2 normalization.
    Returns normalized embeddings.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def save_to_pickle(obj, file_path):
    """
    Saves an object to a file using pickle.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(file_path):
    """
    Loads an object from a pickle file.
    Returns the loaded object.
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
