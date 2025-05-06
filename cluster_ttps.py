import os
import cv2
import numpy as np
from torchvision import transforms, models
import torch
import faiss
import json
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Paths
image_dir = "data/yolo_dataset/images/train"
label_dir = "data/yolo_dataset/labels/train"

def yolo_to_pixels(bbox, img_width, img_height):
    """Convert YOLO bbox (normalized) to pixel coordinates [x1, y1, x2, y2]."""
    x_center, y_center, w, h = bbox
    x1 = int((x_center - w/2) * img_width)
    y1 = int((y_center - h/2) * img_height)
    x2 = int((x_center + w/2) * img_width)
    y2 = int((y_center + h/2) * img_height)
    return [x1, y1, x2, y2]


all_rois = []
all_metadata = []

for img_name in os.listdir(image_dir):
    if not img_name.endswith(('.png')):
        continue
    
    # Read image
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    img_height, img_width = image.shape[:2]
    
    # Read corresponding YOLO label
    label_path = os.path.join(label_dir, img_name.replace('.png', '.txt'))
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        continue 

    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Extract ROIs for each bbox in the image
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])  # Class 0 (thermal pattern)
        bbox = parts[1:5]  # YOLO format: [x_center, y_center, width, height]
        # Convert YOLO bbox to pixel coordinates and crop ROI
        bbox_pixels = yolo_to_pixels(bbox, img_width, img_height)
        roi = image[bbox_pixels[1]:bbox_pixels[3], bbox_pixels[0]:bbox_pixels[2]]
        
        all_rois.append(roi)
        all_metadata.append({
            "image_id": img_name,
            "bbox": bbox_pixels,
            "class": class_id  
        })
    
# Configuration for ROI embeddings 
device = torch.device("cpu")
model = models.efficientnet_b2(weights="IMAGENET1K_V1") 
model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),               
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Extract features for all ROIs
all_features = []

for roi in all_rois:
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    roi_tensor = transform(roi_rgb).unsqueeze(0).to(device)  # Apply transforms and add batch dim
    with torch.no_grad():
        features = model(roi_tensor).flatten().cpu().numpy()
    all_features.append(features)

all_features = np.array(all_features, dtype=np.float32)
# Define FAISS index (L2 distance search)
index = faiss.IndexFlatL2(all_features.shape[1])
# Add feature vectors to FAISS index
index.add(all_features)


similarity_dir = "data/similarity_search"
os.makedirs(similarity_dir, exist_ok=True)
np.save(f"{similarity_dir}/main_features.npy", all_features)
faiss.write_index(index, f"{similarity_dir}/main_index.bin")
with open(f"{similarity_dir}/main_metadata.json", "w") as f:
    json.dump(all_metadata, f)

def assign_damage_and_plot_clusters():
    """
    Performs clustering on feature vectors using KMeans and assigns damage classifications to each ROI.
    
    The function does the following: 
    1. Applies KMeans clustering to group the feature vectors into 5 clusters.
    2. Uses t-SNE for 2D visualization of the clustered data.
    3. Maps each cluster to a predefined damage type (e.g., erosion, crack, etc.).
    4. Plots the clustered data in 2D space.
    """
    with open(f"{similarity_dir}/main_metadata.json", "r") as f:
        main_metadata = json.load(f)
    print(len(all_metadata))
    main_features = np.load(f"{similarity_dir}/main_features.npy")
    print(len(main_features))
    
    kmeans = KMeans(n_clusters=5, random_state=2)
    labels = kmeans.fit_predict(main_features)
    tsne = TSNE(n_components=2, perplexity=40, random_state=2)
    features_2d = tsne.fit_transform(main_features) 
    
    cluster_label_map = {
        0: "erosion",
        1: "delamination",
        2: "crack",
        3: "icing",
        4: "lightning"
    }

    # Update metadata
    for i, meta in enumerate(main_metadata):
        cluster_id = labels[i]
        meta["class"] = cluster_label_map[cluster_id]  

    with open(f"{similarity_dir}/main_metadata.json", "w") as f:
        json.dump(main_metadata, f, indent=4)
        
    # Plot with cluster colors
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, alpha=0.7)
    plt.xlabel("t-SNE_1")
    plt.ylabel("t-SNE_2")
    plt.title("TTP Features")
    plt.show()

assign_damage_and_plot_clusters()