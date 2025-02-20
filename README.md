# Personal Image Retriever

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

<img width="1000" alt="Architecture" src="https://github.com/user-attachments/assets/0f2946de-4393-4dbb-b441-d92d6e2f2213" />


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
  ├── experimentation/
  │   ├── String_Matching_Test.py
  │   ├── Face_Detection_Test.py
  │   ├── Face_Embeddings_test.py
  │   ├── MP_MTCNN_Comp.py
  │   ├── 
  │   └── 
  │ 
  ├── application/   
  │   ├── app.py
  │   ├── store.py
  │   └── utils.py
  │ 
  ├── dataset/  
  │   ├── image_collection
  │   └── reference_images
  │ 
  ├── requirements.txt
  └── README.md
```
## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/AffineAnalytics/personal_image_retriever.git
   cd personal_image_retriever
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit application:
   ```bash
   streamlit run application/app.py
   ```
## Tools and Technologies
* Language: Python
* Framework: PyTorch
* Multimodal Models: OpenAI CLIP, JINA CLIP V2
* Facial Detection Models: MTCNN, InceptionResnet V1
* Vector Store: Facebook AI Similarity Search (FAISS)
* Web Interface: Streamlit

