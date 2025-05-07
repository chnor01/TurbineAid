# Wind Turbine Blade Monitoring using Thermal Images and RAG with LLM 
This project presents a decision support system for detecting and analyzing **thermal turbulence patterns (TTPs)** on wind turbine blades using **YOLOv11**, **K-Means clustering**, **FAISS similarity search**, and **LLM-based Retrieval-Augmented Generation (RAG)**. It aims to automate fault detection and provide interpretable, context-based recommendations for wind turbine maintenance.

## 📌 Features

- **YOLOv11 Object Detection**  
  Detects thermal turbulence patterns (TTPs) in thermal images with a mean Average Precision (mAP@0.5) of **81.1%**.

- **TTP Clustering**  
  K-Means clustering groups TTPs into 5 visual categories; clusters are manually assigned damage types (crack, erosion, delamination, lightning damage, icing damage).

- **FAISS Similarity-Based Classification**  
  New TTP detections are matched to similar clustered features and classified based on majority vote.

- **RAG LLM System**  
  A Retrieval-Augmented Generation (RAG) pipeline retrieves relevant content from technical documentation based on the detected damage type and the user's query. This context is passed to a locally hosted LLM (Gemma 3 4B) to generate informed, domain-specific responses.

- **OpenWebUI Integration**  
  A chat interface allows uploading thermal images, detecting damage, retrieving documentation, and receiving the LLM response all in one interaction.

### Project Demo
https://github.com/user-attachments/assets/c7ab91ea-c0ba-4f14-be90-fbe94b1fafe8

