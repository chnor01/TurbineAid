# Standard library imports
import base64
import io
import json
import os
from collections import Counter
from typing import Generator, Iterator, List, Union
from io import BytesIO
from PIL import Image

# Third-party library imports
import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from chromadb import chromadb
from llama_index.core import PromptTemplate, StorageContext
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from matplotlib.patches import Rectangle
from pydantic import BaseModel
from torchvision import models, transforms
from ultralytics import YOLO


class RAGStringQueryEngine(CustomQueryEngine):
    """
    A custom query engine that performs retrieval-augmented generation (RAG) 
    using a string query and optional damage context.
    """
    retriever: BaseRetriever
    llm: Ollama  
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str, damage_response_types: str = None):
        if damage_response_types:
            query_str = f"Based on the damage ({damage_response_types}), {query_str}"
        
        print("query_str:", query_str)
            
        nodes = self.retriever.retrieve(query_str)
        print(f"Retrieved {len(nodes)} chunks")
        
        chunks_with_sources = []
        for node in nodes:
            content = node.node.get_content()
            metadata = node.node.metadata
            doc_name = metadata.get("file_name", "N/A")
            page = metadata.get("page_label", "N/A")
            chunks_with_sources.append(f"{content}\n\n[Source: {doc_name}, Page {page}]")
        context_str = "\n\n".join(chunks_with_sources)
        
        prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(prompt)
        return str(response)
    

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
         self.name = "WindTurbineAid"       
         self.documents = None
         self.index = None
         self.query_engine = None  
         
         self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "gemma3:4b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "bge-m3:latest"),
            }
        )

    async def on_startup(self):
        CHROMA_PATH = "chromadb"
        os.makedirs(CHROMA_PATH, exist_ok=True)
        DATA_PATH = "../data/pdfs"
        
        # Set embedding and LLM via Ollama
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        self.llm = Settings.llm

        self.documents = SimpleDirectoryReader(DATA_PATH).load_data()
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
        split_docs = splitter.get_nodes_from_documents(self.documents)

        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = chroma_client.get_or_create_collection("llama_index_chroma")
        print(f"Chroma collection count: {chroma_collection.count()}")
        
        if chroma_collection.count() > 0:
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
        else:
            documents = SimpleDirectoryReader(DATA_PATH).load_data()
            splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.get_nodes_from_documents(documents)

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(split_docs, storage_context=storage_context)

        self.retriever = index.as_retriever(similarity_top_k=10) # Converts index into a retriever that can fetch the top 10 most similar results

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        
        latest_image_url = None
        messages = body.get("messages", [])
        if messages:  
            latest_message = messages[-1]
            for content in latest_message.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image_url":
                    latest_image_url = content["image_url"]["url"]
                    print(latest_image_url[:100])
                    break 

        if latest_image_url:
            BASE_RESULTS_PATH =  "results"
            os.makedirs(BASE_RESULTS_PATH, exist_ok=True)
            
            existing_runs = [d for d in os.listdir(BASE_RESULTS_PATH) if os.path.isdir(os.path.join(BASE_RESULTS_PATH, d)) and d.startswith("run")]
            run_numbers = [int(d[3:]) for d in existing_runs if d[3:].isdigit()]
            next_run_number = max(run_numbers, default=0) + 1

            # Create the directory for this run
            RESULTS_PATH = os.path.join(BASE_RESULTS_PATH, f"run{next_run_number}")
            os.makedirs(RESULTS_PATH)

            IMAGES_PATH = os.path.join(RESULTS_PATH, "images")
            os.makedirs(IMAGES_PATH)
            LABELS_PATH = os.path.join(RESULTS_PATH, "labels")
            os.makedirs(LABELS_PATH)
            
            
            if latest_image_url.startswith("data:image"):
                header, base64_data = latest_image_url.split(",", 1)
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_bytes))
                original_img = "original.png"
                original_img_path = os.path.join(IMAGES_PATH, original_img)
                image.save(original_img_path)
                            
            model = YOLO("../yolo_model/train/weights/best.pt")
            results = model(original_img_path)
            
            for result in results:            
                if result.boxes and result.boxes.xywhn is not None:
                    boxes = result.boxes.xywhn.numpy()
                    classes = result.boxes.cls.numpy()
                    
                    predicted_img = "predicted.png"
                    predicted_img_path = os.path.join(IMAGES_PATH, predicted_img)
                    result.save(filename=predicted_img_path)
                    
                    label_filename = "predicted.txt"
                    label_save_path = os.path.join(LABELS_PATH, label_filename)

                    with open(label_save_path, "w") as f:
                        for cls, (x, y, w, h) in zip(classes, boxes):
                            f.write(f"{int(cls)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n")
                else:
                    return "No thermal patterns found in image."
                            
            run_numbers = [int(d[3:]) for d in os.listdir(BASE_RESULTS_PATH)
                if d.startswith("run") and d[3:].isdigit()]
            last_run = max(run_numbers) if run_numbers else None
            LAST_RUN_LABELS_PATH = f"results/run{last_run}/labels"

            def yolo_to_pixels(bbox, img_width, img_height):
                """Convert YOLO bbox (normalized) to pixel coordinates [x1, y1, x2, y2]."""
                x_center, y_center, w, h = bbox
                x1 = int((x_center - w/2) * img_width)
                y1 = int((y_center - h/2) * img_height)
                x2 = int((x_center + w/2) * img_width)
                y2 = int((y_center + h/2) * img_height)
                return [x1, y1, x2, y2]
                
            all_rois_cropped = []
            all_pred_metadata = []

            read_image = cv2.imread(original_img_path)
            if read_image is None:
                print(f"Failed to read image: {original_img_path}")
                return "Failed to read image"
            img_height, img_width = read_image.shape[:2]
            
            pred_bbox_txt = os.path.join(LAST_RUN_LABELS_PATH, "predicted.txt")
            with open(pred_bbox_txt, "r") as f:
                lines = f.readlines()
            
            # Extract ROIs for each bbox in the image
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                bbox = parts[1:5]  # Retrieve all coords: [x_center, y_center, width, height]
                bbox_pixels = yolo_to_pixels(bbox, img_width, img_height) # Convert to absolute pixels
                roi_cropped = read_image[bbox_pixels[1]:bbox_pixels[3], bbox_pixels[0]:bbox_pixels[2]] #Crop ROIs
                
                all_rois_cropped.append(roi_cropped)
                all_pred_metadata.append({
                    "image_id": "original.png",
                    "bbox": bbox_pixels,
                    "class": class_id  
                })

            print("Number of ROIs:", len(all_rois_cropped))
            for i in range(len(all_rois_cropped)):
                print(all_rois_cropped[i].shape)
            
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = models.efficientnet_b2(weights="IMAGENET1K_V1") 
            model.classifier = torch.nn.Identity()
            model.eval()

            transform = transforms.Compose([                        
                transforms.ToTensor(),               
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
            ])


            # Extract features from ROIs
            pred_features = []
            for roi in all_rois_cropped:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                roi_tensor = transform(roi_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model(roi_tensor).flatten().cpu().numpy()
                pred_features.append(features)


            pred_features = np.array(pred_features, dtype=np.float32)
            print("Embedding shape:", pred_features.shape)
            # Define FAISS index (L2 distance search)
            #index = faiss.IndexFlatL2(pred_features.shape[1])
            # Add feature vectors to FAISS index
            #index.add(pred_features)
            # Save index to disk

            # Save indices, features, metadata
            #faiss.write_index(index, os.path.join(RESULTS_PATH, "faiss_index_pred.bin"))
            #np.save(os.path.join(RESULTS_PATH, "features_pred.npy"), pred_features)
            with open(os.path.join(RESULTS_PATH, "metadata_pred.json"), "w") as f:
                json.dump(all_pred_metadata, f)
                
                
            run_numbers = [int(d[3:]) for d in os.listdir(BASE_RESULTS_PATH) if d.startswith("run") and d[3:].isdigit()]
            last_run = max(run_numbers) if run_numbers else None

            #pred_features = np.load(f"{BASE_RESULTS_PATH}/run{last_run}/features_pred.npy")
            index = faiss.read_index("../data/similarity_search/main_index.bin")
            with open("../data/similarity_search/main_metadata.json", "r") as f:
                main_metadata = json.load(f)
            
            predicted_damage_types = []
            k = 5  # Number of nearest neighbors
            distances, indices = index.search(pred_features, k)

            print("Top 5 similar images:", indices)
            print("Distances:", distances)
            for query_idx in range(len(pred_features)):
                neighbor_types = []
                print(f"\nQuery ROI {query_idx}:")

                for i in range(k):
                    idx = indices[query_idx][i] 
                    neighbor_metadata = main_metadata[idx]
                    damage_type = neighbor_metadata.get("class")
                    dist = distances[query_idx][i]
                    neighbor_types.append(damage_type)
                    #print(f"Neighbor {i+1}: {neighbor_metadata}, Distance: {dist:.2f}")

                type_counts = Counter(neighbor_types)
                most_common_type, count = type_counts.most_common(1)[0] # Count most common damage type
                if count >= 4:
                    predicted_type = most_common_type
                else:
                    predicted_type = "unknown"

                predicted_damage_types.append(predicted_type)
                print(f"Query {query_idx}: predicted damage type: {predicted_type}")
                
                
            all_damage_class = []
            with open(f"{BASE_RESULTS_PATH}/run{last_run}/metadata_pred.json", "r+") as f:
                pred_metadata = json.load(f)
                for i in range(len(pred_metadata)):
                    pred_metadata[i]["class"] = predicted_damage_types[i]
                    all_damage_class.append(predicted_damage_types[i])
                f.seek(0)
                json.dump(pred_metadata, f, indent=4)
                f.truncate()
                
            damage_count = Counter(all_damage_class) # Count all damage types (lightning, crack, etc.) and their respective quantity
            damage_response = "\n".join([f"{dmg}: {qty}" for dmg, qty in damage_count.items()]) # Format response for prompt
            print(damage_response)
            damage_response_types = ", ".join([f"{dmg}" for dmg in damage_count.keys()])
                
            def annotate_damage_classes(images_dir, json_path):
                label_colors = {
                "erosion": "blue", 
                "lightning": "red",
                "ice": "cyan",
                "crack": "green",
                "delamination": "purple",
                "unknown": "yellow"
                }
                
                # Load image
                img = cv2.imread(f"{images_dir}/original.png")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                
                    # Create a plot
                fig, ax = plt.subplots(1, figsize=(12, 8))
                ax.imshow(img_rgb)
                
                with open(json_path, "r") as f:
                        pred_data = json.load(f)

                for roi in pred_data:
                    bbox_coords = roi["bbox"]  
                    damage_type = roi["class"]
                    
                    x_min, y_min, x_max, y_max = bbox_coords
                    
                    color = label_colors.get(damage_type)
                    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor="none")
                    ax.add_patch(rect)
                    ax.text(x_min, y_min - 10, damage_type[:3].upper(), color=color, fontsize=12, fontweight="bold")


                ax.axis("off")
                plt.savefig(f"{images_dir}/predicted_dmg_types.png", bbox_inches="tight", pad_inches=0)
                plt.close()
            
            annotate_damage_classes(f"{BASE_RESULTS_PATH}/run{last_run}/images", 
                                    f"{BASE_RESULTS_PATH}/run{last_run}/metadata_pred.json")
            
            
            def create_markdown_img(img_path):
                img = Image.open(img_path)
                img = img.resize((320, 256))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_url = f"data:image/png;base64,{img_b64}"
                image_data = f"\n![image]({image_url})\n"
                return image_data
            
            markdown_img = create_markdown_img(f"{BASE_RESULTS_PATH}/run{last_run}/images/predicted_dmg_types.png")
        else:
            print("No image uploaded, skipping to RAG...")
            
        qa_prompt = PromptTemplate(
            """You are an expert wind turbine technician assistant. Your task is to provide accurate, 
            concise, and practical answers based ONLY on the technical context provided below.

            If the answer cannot be found in the context, reply with: 
            "I couldn't find a reliable answer in the provided PDF documents."

            Do not guess or add information not present in the context.

            ---
            Relevant Technical Context:
            {context_str}
            ---

            **Question:** {query_str}

            Provide your answer below. Use bullet points if appropriate, and cite sources using this format: 
            [Source: filename, page X].
            """
        )

        
        query_engine = RAGStringQueryEngine(
            retriever=self.retriever,
            llm=self.llm,
            qa_prompt=qa_prompt,
        )

        if latest_image_url: # Response if user uploads an image
            response = query_engine.custom_query(user_message, damage_response_types)
            final_response = response + "\n" + markdown_img # RAG response + predicted image with damage classifications
            print(response)
            return final_response
        
        response = query_engine.custom_query(user_message) # RAG LLM Response only
        print(response)
        return response
            
        
        
        
        
        
