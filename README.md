# SnapQuery: Your Personalized Image Search Assistant

_**Powered by Qualcomm AI Hub**_

## Introduction
With the increasing volume of digital images, finding specific photos in large, unorganized collections can be a tedious task. This project addresses this problem by aiming to create a smart and intuitive image retrieval system that allows users to find images based on text queries and facial recognition. The system leverages multimodal models, vector stores, and semantic search techniques to provide accurate and efficient image retrieval.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Folder Structure](#folder-structure)
- [Installation and Usage](#installation-and-usage)
- [Tools and Technologies](#tools-and-technologies)

## Features
- **Text-Based Search**: Find images by describing them in your own words.
- **Facial Recognition**: Locate images featuring specific individuals.
- **Efficient Retrieval**: Optimized indexing and ranking for quick results.
- **Scalable Design**: Built to handle large collections of images.

## How It Works

<img width="1046" alt="Architecture" src="https://github.com/user-attachments/assets/a96dd9cf-1651-4691-9d6a-4c9884c1c1fa" />



1. **Preprocessing**: Images are processed to extract text descriptions and facial features.
2. **Indexing**: Extracted information is stored in separate indexes for quick retrieval.
3. **Query Execution**: Users enter a search query, which is analyzed for text and facial references.
4. **Matching**: The system compares the query with indexed data to find relevant images.
5. **Ranking**: Results are ranked based on similarity and displayed to the user.

## Folder Structure
Here is the structure of the project:

```plaintext
  <Personal Image Retriever>/
  │
  ├── application/   
  │   ├── app.py
  │   ├── store.py
  │   └── utils.py
  │ 
  ├── assets/
  │   ├── CLIP_faiss_index.index
  │   ├── face_faiss_index.index
  │   ├── img_paths.pkl
  │   └── ref_emb.pkl
  │
  ├── dataset/  
  │   ├── image_collection
  │   └── reference_images
  │
  ├── experimentation/
  │   ├── String_Matching_Test.py
  │   ├── Face_Detection_Test.py
  │   ├── Face_Embeddings_test.py
  │   ├── MP_MTCNN_Comp.py
  │   └── application_v1/
  │       ├── app_v1.py
  │       ├── store_v1.py
  │       └── utils_v1.py
  │ 
  ├── README.md
  ├── packages.txt
  └── requirements.txt
```
## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/AffineAnalytics/personal_image_retriever.git
   cd personal_image_retriever
   ```
2. Configure AI Hub Access:
   -  [Create a Qualcomm® ID](https://myaccount.qualcomm.com/signup), and use it to [login to Qualcomm® AI Hub](https://app.aihub.qualcomm.com/).
   -  Configure your [API token](https://app.aihub.qualcomm.com/account/) and replace the placeholder API_TOKEN below with it
   ```bash
   pip install qai-hub
   qai-hub configure --api_token API_TOKEN
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Streamlit application:
   ```bash
   streamlit run application/app.py
   ```
5. Test your own dataset (Optional):
   -  If you want to test it out on your own image dataset, replace the images in the [dataset/image_collection](https://github.com/AffineAnalytics/personal_image_retriever/tree/main/dataset/image_collection) folder with your dataset in JPG, JPEG or PNG format.
   -  Upload your own reference images (with their names as the filename) to [dataset/reference_images](https://github.com/AffineAnalytics/personal_image_retriever/tree/main/dataset/reference_images) in JPG, JPEG or PNG format.
   -  Empty the [assets]([https://github.com/AffineAnalytics/personal_image_retriever/tree/31084b570d56876bc4a010035ce0757ac970c741/assets](https://github.com/AffineAnalytics/personal_image_retriever/tree/main/assets)) folder and then run the below:
   ```bash
   python application/store.py
   streamlit run application/app.py
   ```  

## Tools and Technologies
* Language: Python
* Framework: PyTorch
* Qualcomm AI Hub Models: OpenAI CLIP (Image and Text Encoding), MediapPipe Face (Face Recognition)
* Vector Store: Facebook AI Similarity Search (FAISS)
* Web Interface: Streamlit

