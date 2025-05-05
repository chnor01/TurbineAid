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
        """
        Executes a query using retrieved documents and a language model.

        Args:
            query_str (str): The user's query.
            damage_response_types (str, optional): Damage type(s) to be included in the query context.

        Returns:
            str: The LLM-generated response based on retrieved context.
        """
        
        # Incorporate damage information into the query if provided
        if damage_response_types:
            query_str = f"Based on the damage ({damage_response_types}), {query_str}"
        print("query_str:", query_str)
            
        # Retrieve relevant document nodes based on the query
        nodes = self.retriever.retrieve(query_str)
        print(f"Retrieved {len(nodes)} chunks")
        
        # Assemble retrieved content with source metadata
        chunks_with_sources = []
        for node in nodes:
            content = node.node.get_content()
            metadata = node.node.metadata
            doc_name = metadata.get("file_name", "N/A")
            page = metadata.get("page_label", "N/A")
            chunks_with_sources.append(f"{content}\n\n[Source: {doc_name}, Page {page}]")
        context_str = "\n\n".join(chunks_with_sources)
        
        # Format the final prompt and generate the LLM response
        prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        response = self.llm.complete(prompt)
        return str(response)
    

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str # URL for the Ollama server
        LLAMAINDEX_MODEL_NAME: str # Name of the LLM used for query completion
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str # Name of the embedding model used for document embeddings

    def __init__(self):
         self.name = "WindTurbineAid"       
         self.documents = None
         self.index = None
         self.query_engine = None  
         
         # Load configuration from environment variables
         self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "gemma3:4b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "bge-m3:latest"),
            }
        )

    async def on_startup(self):
        """
        Initializes the ChromaDB vector store, loads documents, and sets up the retriever when the Open WebUI's Pipelines server starts.

        This function does the following:
        1. Sets up the embedding model and LLM using Ollama.
        2. Initializes a ChromaDB client and checks if documents are already indexed.
        3. If no documents exist, it loads PDFs, splits them into chunks, and indexes them.
        4. If documents are already indexed, it loads the existing vector store.
        5. Initializes a retriever for similarity search.
        """
        
        CHROMA_PATH = "chromadb"
        os.makedirs(CHROMA_PATH, exist_ok=True)
        DATA_PATH = "../data/pdfs"
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data path {DATA_PATH} does not exist")
        
        try:
            # Configure Ollama embedding and LLM models
            Settings.embed_model = OllamaEmbedding(
                model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            Settings.llm = Ollama(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            self.llm = Settings.llm

            # Initialize Chroma client and collection
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            chroma_collection = chroma_client.get_or_create_collection("llama_index_chroma")
            print(f"Chroma collection count: {chroma_collection.count()}")
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
            
            # Load, chunk, index the documents if the collection is empty
            if chroma_collection.count() == 0:
                documents = SimpleDirectoryReader(DATA_PATH).load_data()
                split_docs = splitter.get_nodes_from_documents(documents)

                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex(split_docs, storage_context=storage_context)
            else:
                # Load ChromaDB index if it exists
                index = VectorStoreIndex.from_vector_store(vector_store)
                
            # Create a retriever to fetch top-10 most similar chunks for a given query
            self.retriever = index.as_retriever(similarity_top_k=10)
            
        except Exception as e:
            print(f"Error during startup: {str(e)}")
            raise

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process a user message containing an image (optional) and generate a response using either:
        - Image analysis pipeline (if image is provided)
        - RAG-LLM response (if no image is provided)
    
        The image analysis pipeline performs:
        1. Thermal turbulence detection (TTP) using YOLOv11
        2. Extraction of TTP regions and feature embedding
        3. Damage type classification using similarity search
        4. Visual annotation of classified damage types
        5. Combined response with RAG results
    
        Processing Flow:
            1. Check for image in latest message to any Open WebUI chat
            2. If image exists:
                a. Save original image
                b. Run YOLO detection for thermal patterns
                c. Extract ROIs and convert to embeddings
                d. Classify damage types using FAISS similarity search
                e. Annotate image with damage classifications
                f. Combine RAG response with damage classifications from image
            3. If no image:
                a. Use only user query for RAG LLM response
        """
        
        # Retrieve base64 encoding of user's image if it exists
        latest_image_url = None
        messages = body.get("messages", [])
        if messages:  
            latest_message = messages[-1]
            for content in latest_message.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image_url":
                    latest_image_url = content["image_url"]["url"]
                    break 
        
        # Create directories for results
        if latest_image_url:
            BASE_RESULTS_PATH =  "results"
            os.makedirs(BASE_RESULTS_PATH, exist_ok=True)
            
            run_numbers = sorted([int(d[3:]) for d in os.listdir(BASE_RESULTS_PATH) if d.startswith("run") and d[3:].isdigit()])
            last_run = (run_numbers[-1] + 1) if run_numbers else 1

            RESULTS_PATH = os.path.join(BASE_RESULTS_PATH, f"run{last_run}")
            IMAGES_PATH = os.path.join(RESULTS_PATH, "images")
            LABELS_PATH = os.path.join(RESULTS_PATH, "labels")
            
            os.makedirs(RESULTS_PATH)
            os.makedirs(IMAGES_PATH)
            os.makedirs(LABELS_PATH)
            
            # Save original image
            if latest_image_url.startswith("data:image"):
                header, base64_data = latest_image_url.split(",", 1)
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_bytes))
                original_img = "original.png"
                original_img_path = os.path.join(IMAGES_PATH, original_img)
                image.save(original_img_path)
                           
            # Run inference on image 
            model = YOLO("../yolo_model/train/weights/best.pt")
            results = model(original_img_path)
            
            # Save predicted image and predicted bounding box coordinates if they exist
            for result in results:            
                if result.boxes and result.boxes.xywhn is not None:
                    boxes = result.boxes.xywhn.numpy()
                    classes = result.boxes.cls.numpy()
                    
                    predicted_img = "predicted.png"
                    predicted_img_path = os.path.join(IMAGES_PATH, predicted_img)
                    result.save(filename=predicted_img_path)
                    
                    predicted_labels = "predicted.txt"
                    predicted_labels_path = os.path.join(LABELS_PATH, predicted_labels)

                    with open(predicted_labels_path, "w") as f:
                        for cls, (x, y, w, h) in zip(classes, boxes):
                            f.write(f"{int(cls)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n")
                else:
                    return "No thermal patterns found in image."
                
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
            
            pred_bbox_txt = os.path.join(LABELS_PATH, "predicted.txt")
            with open(pred_bbox_txt, "r") as f:
                lines = f.readlines()
            
            # Extract ROIs for each bounding box in the image
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                bbox = parts[1:5]
                bbox_pixels = yolo_to_pixels(bbox, img_width, img_height)
                roi_cropped = read_image[bbox_pixels[1]:bbox_pixels[3], bbox_pixels[0]:bbox_pixels[2]]
                
                all_rois_cropped.append(roi_cropped)
                all_pred_metadata.append({
                    "image_id": "original.png",
                    "bbox": bbox_pixels,
                    "class": class_id  
                })
            with open(os.path.join(RESULTS_PATH, "pred_metadata.json"), "w") as f:
                json.dump(all_pred_metadata, f)

            # Configure EfficientNet-B2 for ROI embeddings
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
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_tensor = transform(roi_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model(roi_tensor).flatten().cpu().numpy()
                pred_features.append(features)
            pred_features = np.array(pred_features, dtype=np.float32)
            
            # Load main index and metadata for similarity search
            index = faiss.read_index("../data/similarity_search/main_index.bin")
            with open("../data/similarity_search/main_metadata.json", "r") as f:
                main_metadata = json.load(f)
            
            predicted_damage_types = []
            k = 5  # Number of nearest neighbors
            
            # Perform similarity search on ROIs
            distances, indices = index.search(pred_features, k)
            print("Top 5 similar images:", indices)
            print("Distances:", distances)
            
            # Determine damage type based on majority vote from most similar ROIs
            for query_idx in range(len(pred_features)):
                neighbor_types = []
                print(f"\nQuery ROI {query_idx}:")

                for i in range(k):
                    idx = indices[query_idx][i] 
                    neighbor_metadata = main_metadata[idx]
                    damage_type = neighbor_metadata.get("class")
                    neighbor_types.append(damage_type)

                type_counts = Counter(neighbor_types)
                most_common_type, count = type_counts.most_common(1)[0]
                if count >= 4:
                    predicted_type = most_common_type
                else:
                    predicted_type = "unknown"

                predicted_damage_types.append(predicted_type)
                print(f"Query {query_idx}: predicted damage type: {predicted_type}")
                
            # Save metadata
            all_damage_class = []
            with open(f"{RESULTS_PATH}/pred_metadata.json", "r+") as f:
                pred_metadata = json.load(f)
                for i in range(len(pred_metadata)):
                    pred_metadata[i]["class"] = predicted_damage_types[i]
                    all_damage_class.append(predicted_damage_types[i])
                f.seek(0)
                json.dump(pred_metadata, f, indent=4)
                f.truncate()
                
            # Format damage types for RAG input query
            damage_count = Counter(all_damage_class)
            #damage_response = "\n".join([f"{dmg}: {qty}" for dmg, qty in damage_count.items()])
            damage_response_types = ", ".join([dmg for dmg in damage_count.keys() if dmg != "unknown"]) # Skip 'unknown' damage
                
            def annotate_damage_types(images_dir, json_path):
                """
                Annotates and saves an image with bounding boxes and labels for different types of damage.
                """
                
                label_colors = {
                "erosion": "blue", 
                "lightning": "red",
                "icing": "cyan",
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

                # Extract bounding box coordinates and corresponding damage types
                for roi in pred_data:
                    bbox_coords = roi["bbox"]  
                    damage_type = roi["class"]
                    
                    x_min, y_min, x_max, y_max = bbox_coords
                    
                    # Draw colored bounding boxes with text labels based on damage type. Text labels show first 3 characters of damage type (e.g. "ERO" for "erosion")
                    color = label_colors.get(damage_type)
                    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor="none")
                    ax.add_patch(rect)
                    ax.text(x_min, y_min - 10, damage_type[:3].upper(), color=color, fontsize=12, fontweight="bold")

                ax.axis("off")
                plt.savefig(f"{images_dir}/predicted_dmg_types.png", bbox_inches="tight", pad_inches=0)
                plt.close()
            
            annotate_damage_types(IMAGES_PATH, 
                                    f"{RESULTS_PATH}/pred_metadata.json")
            
            
            def create_markdown_img(img_path):
                """
                Converts an image to a base64-encoded markdown image for display in the chat interface
                """
                img = Image.open(img_path)
                img = img.resize((352, 282))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_url = f"data:image/png;base64,{img_b64}"
                image_data = f"\n![image]({image_url})\n"
                return image_data
            
            markdown_img = create_markdown_img(f"{IMAGES_PATH}/predicted_dmg_types.png")
            
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

        # Response if user uploads an image. Includes RAG LLM response + markdown image
        if latest_image_url:
            response = query_engine.custom_query(user_message, damage_response_types)
            final_response = response + "\n" + markdown_img
            print(response)
            return final_response
        
        # RAG LLM response only.        
        response = query_engine.custom_query(user_message)
        print(response)
        return response
            
        
        
        
        
        
