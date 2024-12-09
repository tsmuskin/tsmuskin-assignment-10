import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from open_clip import create_model_and_transforms, get_tokenizer
import pandas as pd
from sklearn.decomposition import PCA

# Initialize the model and preprocessing (done once, globally for performance)
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')

# Load the precomputed image embeddings DataFrame (done once, globally for performance)
df = pd.read_pickle('image_embeddings.pickle')
embeddings = torch.stack([torch.tensor(e) for e in df['embedding'].values])
embeddings = F.normalize(embeddings, p=2, dim=1)  # Ensure embeddings are normalized

# Assume these variables are initialized globally in the module
pca = None
pca_embeddings = None
current_k_components = None

def initialize_pca(k_components, embeddings):
    """
    Initialize or update PCA with the specified number of components.
    """
    global pca, pca_embeddings, current_k_components

    print(f"Initializing PCA: Current={current_k_components}, Requested={k_components}")
    if pca is None or current_k_components != k_components:
        print("Recomputing PCA embeddings...")
        current_k_components = k_components
        pca = PCA(n_components=k_components)
        pca_embeddings = pca.fit_transform(embeddings.numpy())
        pca_embeddings = F.normalize(torch.tensor(pca_embeddings), p=2, dim=1)
        print(f"PCA embeddings updated with shape: {pca_embeddings.shape}")


def search_with_image(image_path, use_pca, k_components):
    """
    Searches for the most similar images to the query image.

    Args:
        image_path (str): Path to the query image.

    Returns:
        list: A list of top 5 similar images and their similarity scores.
    """
    global pca_embeddings
    # Preprocess the input image
    image = preprocess(PILImage.open(image_path)).unsqueeze(0)
    
    # Compute query embedding
    query_embedding = F.normalize(model.encode_image(image), p=2, dim=1)

    if use_pca:
        # Ensure PCA embeddings are updated with the desired number of components
        initialize_pca(k_components, embeddings)
       
        # Reduce query embedding dimensions using PCA
        query_pca_embedding = pca.transform(query_embedding.detach().numpy())
        query_pca_embedding = F.normalize(torch.tensor(query_pca_embedding), p=2, dim=1)

        similarities = torch.mm(query_pca_embedding, pca_embeddings.T).squeeze(0)
        
        top_scores, top_indices = torch.topk(similarities, k=5)
        print("Top Scores:", top_scores)
        print("Top Indices: ", top_indices)

        # Map indices back to file names and scores
        results = [
            (df.iloc[idx]['file_name'], score)
            for idx, score in zip(top_indices.tolist(), top_scores.tolist())
        ]
        return results

    else:
        # Compute cosine similarity between the query embedding and dataset embeddings
        similarities = torch.mm(query_embedding, embeddings.T).squeeze(0)

        # Get the top 5 most similar images and their indices
        top5_indices = torch.topk(similarities, 5).indices.tolist()
        top5_scores = torch.topk(similarities, 5).values.tolist()

        # Retrieve paths to the top 5 images
        results = []
        for idx, score in zip(top5_indices, top5_scores):
            impath = df.iloc[idx]['file_name']
            results.append((impath, score))
        return results

def search_with_text(query_text):
    """
    Perform a text-to-image search based on the provided query text.
    
    Args:
        query_text (str): The text query for searching.

    Returns:
        list: A list of dictionaries containing image paths and similarity scores.
    """
    # Tokenize and encode the query text
    text = tokenizer([query_text])
    query_embedding = F.normalize(model.encode_text(text), p=2, dim=1)

    # Compute cosine similarity
    similarities = torch.mm(query_embedding, embeddings.T).squeeze(0)

    # Retrieve top 5 most similar images
    top_indices = torch.topk(similarities, k=5).indices.tolist()
    top_scores = similarities[top_indices].tolist()
    results = []

    for idx, score in zip(top_indices, top_scores):
        impath = df.iloc[idx]['file_name']
        results.append((impath, score))

        # results.append({
        #     'image': impath,
        #     'score': score
        # })

    return results

def search_combined(query_text, image_path, weight, use_pca, k_components):
    global pca_embeddings


    image = preprocess(PILImage.open(image_path)).unsqueeze(0)
    text = tokenizer([query_text])

    image_query = F.normalize(model.encode_image(image),p=2,dim=1)
    text_query = F.normalize(model.encode_text(text),p=2,dim=1)

    combined_query = weight * text_query + (1.0 - weight) * image_query
    
    if use_pca:
        # Ensure PCA embeddings are updated with the desired number of components
        initialize_pca(k_components, embeddings)

        query_pca_embedding = pca.transform(combined_query.detach().numpy())
        query_pca_embedding = F.normalize(torch.tensor(query_pca_embedding), p=2, dim=1)


        similarities = torch.mm(query_pca_embedding, pca_embeddings.T).squeeze(0)
        
        top_scores, top_indices = torch.topk(similarities, k=5)
        print("Top Scores:", top_scores)
        print("Top Indices: ", top_indices)

        # Map indices back to file names and scores
        results = [
            (df.iloc[idx]['file_name'], score)
            for idx, score in zip(top_indices.tolist(), top_scores.tolist())
        ]
        return results

    else:
        combined_query = F.normalize(combined_query, p=2,dim=1)
        similarities = torch.mm(combined_query, embeddings.T).squeeze(0)

        top_indices = torch.topk(similarities, k=5).indices.tolist()
        top_scores = similarities[top_indices].tolist()
        results = []

        for idx, score in zip(top_indices, top_scores):
            impath = df.iloc[idx]['file_name']
            results.append((impath, score))
        return results
