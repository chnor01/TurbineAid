import os
import numpy as np
import cv2
import json
from skimage import exposure
from skimage.filters import unsharp_mask
import shutil
import random


def npy_to_image(npy_array):
    """
    Converts a NumPy array (wind turbine blade thermograms) into a grayscale image. Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) 
    for contrast enhancement and unsharp masking for sharpness. Based on code from KI-VISIR dataset, 'ki_visir_helper_functions_v2.py'.
    """
    # Calculate mean without NaNs
    mean_val = np.nanmean(npy_array)
    npy_array[npy_array < mean_val] = np.nan
  
    # Mask out values below the mean by setting them to NaN
    min_val = np.nanpercentile(npy_array, 5)
    max_val = np.nanpercentile(npy_array, 95)
    
    # Compute the 5th and 95th percentiles
    normalized_image = (npy_array - min_val) / (max_val - min_val)
    normalized_image = np.clip(normalized_image, 0, 1).astype(np.float32)

    # Stack the array into 3 channels for compatibility
    min_valid = np.nanmin(normalized_image)
    normalized_image = np.where(np.isnan(normalized_image), min_valid, normalized_image)
    normalized_image = np.stack((normalized_image,) * 3, axis=-1)
   
    # Apply CLAHE and unsharp masking
    clahe_img = exposure.equalize_adapthist(normalized_image, clip_limit=0.01)
    unsharp_clahe_img = unsharp_mask(clahe_img, radius=4, amount=1.5)
    unsharp_clahe_img = np.clip(unsharp_clahe_img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite("data/xdd.png", unsharp_clahe_img)
    return unsharp_clahe_img


def convert_npy_folder_to_png(input_folder, output_folder):
    """
    Saves images generated from 'npy_to_image()' as .png files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            npy_path = os.path.join(input_folder, filename)
            npy_array = np.load(npy_path)
            image = npy_to_image(npy_array)

            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, image)


def geojson_to_yolo(geojson):
    """
    Converts .geojson annotations to YOLO-format annotations: [class x_center y_center width height]
    """
    image_width = 640
    image_height = 512
    yolo_data = []
    class_map = {"ttp": 0}
    
    for feature in geojson['features']:
        # Get the coordinates of the polygon (bounding box)
        coords = feature['geometry']['coordinates'][0]

        # Calculate the bounding box from the polygon
        min_x = min(coord[0] for coord in coords)
        max_x = max(coord[0] for coord in coords)
        min_y = min(coord[1] for coord in coords)
        max_y = max(coord[1] for coord in coords)
        
        # Calculate the center and size of the bounding box
        x_center = (min_x + max_x) / 2 / image_width
        y_center = (min_y + max_y) / 2 / image_height
        width = (max_x - min_x) / image_width
        height = (max_y - min_y) / image_height
        
        # Get the class ID from the classification name
        class_id = class_map.get(feature['properties']['classification']['name'], -1)
        
        # Append the YOLO format data
        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
    return yolo_data

def save_yolo_annotations(input_folder, output_folder):
    """
    Saves YOLO-format annotations generated from 'geojson_to_yolo() as .txt files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.endswith(".geojson"):
            # Construct the full file path
            file_path = os.path.join(input_folder, filename)

            # Load the GeoJSON file
            with open(file_path, 'r') as file:
                geojson = json.load(file)
            
            # Convert to YOLO format
            yolo_annotations = geojson_to_yolo(geojson)
            
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            # Save the YOLO annotations as .txt file
            with open(output_file_path, "w") as output_file:
                output_file.write("\n".join(yolo_annotations))
                

def background_images_split(input_annotations, input_images, output_annotated_images, output_background_images):
    """
    Splits images depending on whether they have corresponding annotations or not (empty annotation files).
    """
    os.makedirs(output_annotated_images, exist_ok=True)
    os.makedirs(output_background_images, exist_ok=True)
    
    # Get image base filename
    for file in os.listdir(input_annotations):
        if file.endswith(".txt"):
            file_path = os.path.join(input_annotations, file)
            image_name = os.path.splitext(file)[0] + ".png" 
            image_path = os.path.join(input_images, image_name)

            # Image is "annotated" if corresponding .txt file size isn't empty
            if os.path.exists(image_path):  
                if os.path.getsize(file_path) > 0: 
                    shutil.copy(image_path, os.path.join(output_annotated_images, image_name))
                else: 
                    shutil.copy(image_path, os.path.join(output_background_images, image_name))

def background_annotations_split(input_annotations, output_annotations_background, output_annotations_annotated):
    """
    Splits annotation files depending on whether the files are empty or not. 
    """
    os.makedirs(output_annotations_background)
    os.makedirs(output_annotations_annotated)
    
    # Process annotation files
    for file in os.listdir(input_annotations):
        if file.endswith(".txt"):
            file_path = os.path.join(input_annotations, file)
            
            # Check if file has content
            if os.path.getsize(file_path) > 0:  # Non-empty file
                shutil.copy(file_path, os.path.join(output_annotations_annotated, file))
            else:  # Empty file (background image)
                shutil.copy(file_path, os.path.join(output_annotations_background, file))
                
  
def create_yolo_dataset(input_images_annotated, input_images_background, input_annotations_annotated, input_annotations_background):  
    """
    Creates a YOLO-compatible dataset by splitting annotated and background images into a 75/15/10 split for
    train, val, and test sets. Annotations are included for train/val sets; test set contains images only. 
    All sets (train, val, test) include 10% background images (images with no annotations).
    """
    output_base = "data/yolo_dataset"

    # Create directories
    for path in ["images/train", "images/val", "labels/train", "labels/val", "images/test"]:
        os.makedirs(os.path.join(output_base, path), exist_ok=True)

    # Get image filenames (without extensions)
    image_files = [f for f in os.listdir(input_images_annotated) if f.endswith(".png")]
    image_names = [os.path.splitext(f)[0] for f in image_files]

    image_files_bg = [file for file in os.listdir(input_images_background) if file.endswith(".png")]
    image_names_bg = [os.path.splitext(file)[0] for file in image_files_bg]

    # Shuffle order
    random.shuffle(image_names)
    random.shuffle(image_names_bg)

    # Split into train, val, test sets
    train_split = int(0.75*len(image_names))
    val_split = int(0.15*len(image_names))
    test_split = int(0.10*len(image_names))

    # Include 10% background images in all sets
    bg_train_split = int(0.1*train_split)
    bg_val_split = int(0.1*val_split)
    bg_test_split = int(0.1*test_split)

    # Split
    train_files = image_names[:train_split]
    val_files = image_names[train_split:train_split+val_split]
    test_files = image_names[train_split+val_split:]

    train_files_bg = image_names_bg[:bg_train_split]
    val_files_bg = image_names_bg[bg_train_split:bg_train_split+bg_val_split]
    test_files_bg = image_names_bg[bg_train_split+bg_val_split:bg_train_split+bg_val_split+bg_test_split]

    def move_files(image_list, src_images, dst_images, src_annotations="", dst_annotations="", annotations=True):
        """
        Moves images and annotation files.
        """
        for img_name in image_list:
            img_path = os.path.join(src_images, img_name + ".png")
            ann_path = os.path.join(src_annotations, img_name + ".txt")  # Annotation file

            dst_img = os.path.join(dst_images, img_name + ".png")
            dst_ann = os.path.join(dst_annotations, img_name + ".txt")

            # Move image file
            shutil.move(img_path, dst_img)

            # Move annotation file (skip for test set)
            if os.path.exists(ann_path) and annotations:
                shutil.move(ann_path, dst_ann)

    move_files(train_files, input_images_annotated, os.path.join(output_base, "images/train"), input_annotations_annotated, os.path.join(output_base, "labels/train"))
    move_files(val_files, input_images_annotated, os.path.join(output_base, "images/val"), input_annotations_annotated, os.path.join(output_base, "labels/val"))
    move_files(test_files, input_images_annotated, os.path.join(output_base, "images/test"), annotations=False)
    move_files(train_files_bg, input_images_background,os.path.join(output_base, "images/train"), input_annotations_background, os.path.join(output_base, "labels/train"))
    move_files(val_files_bg, input_images_background, os.path.join(output_base, "images/val"), input_annotations_background, os.path.join(output_base, "labels/val"))
    move_files(test_files_bg, input_images_background, os.path.join(output_base, "images/test"), annotations=False)
    

ki_visir_folder = "data/ki-visir_dataset"
npy_folder = os.path.join(ki_visir_folder, "thermo_npy")
thermo_images_folder = os.path.join(ki_visir_folder, "thermo_png")
thermo_images_annotated = os.path.join(ki_visir_folder, "thermo_png_annotated")
thermo_images_background = os.path.join(ki_visir_folder, "thermo_png_background")
annotations_folder = os.path.join(ki_visir_folder, "thermo_annotations")
yolo_annotations_folder = os.path.join(ki_visir_folder, "yolo_annotations")
yolo_annotations_background = os.path.join(ki_visir_folder, "yolo_annotations_background")
yolo_annotations_annotated = os.path.join(ki_visir_folder, "yolo_annotations_annotated")
#convert_npy_folder_to_png(npy_folder, thermo_images_folder)
#save_yolo_annotations(annotations_folder, yolo_annotations_folder)
#background_images_split(yolo_annotations_folder, thermo_images_folder, thermo_images_annotated, thermo_images_background)
#background_annotations_split(yolo_annotations_folder, yolo_annotations_background, yolo_annotations_annotated)
#create_yolo_dataset(thermo_images_annotated, thermo_images_background, yolo_annotations_annotated, yolo_annotations_background)