from database.mongodb_utils import get_text_embedding, get_collection
from tqdm import tqdm
from typing import List, Dict, Any


def get_documents_without_embeddings(limit: int = None) -> List[Dict[str, Any]]:
    """
    Get documents that have descriptions but no embeddings.
    
    Args:
        limit (int, optional): Maximum number of documents to return. Defaults to None.
    
    Returns:
        List[Dict[str, Any]]: List of documents
    """
    collection = get_collection()
    query = {
        "description": {"$exists": True, "$ne": None},
        "description_embedding": {"$exists": False}  # Changed back to check if field doesn't exist
    }
    
    if limit:
        return list(collection.find(query).limit(limit))
    return list(collection.find(query))

def generate_embeddings_for_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for descriptions in documents.
    
    Args:
        documents (List[Dict[str, Any]]): List of documents to process
    
    Returns:
        List[Dict[str, Any]]: List of documents with their embeddings
    """
    documents_with_embeddings = []
    
    for doc in documents:
        description = doc['description']
        if description:
                       
            embedding = get_text_embedding(description)
            if embedding:
                doc['description_embedding'] = embedding
                documents_with_embeddings.append(doc)
    
    return documents_with_embeddings

def update_documents_with_embeddings(documents: List[Dict[str, Any]]) -> int:
    """
    Update documents in the collection with their embeddings.
    
    Args:
        documents (List[Dict[str, Any]]): List of documents with embeddings
    
    Returns:
        int: Number of documents updated
    """
    if not documents:
        return 0
        
    collection = get_collection()
    updated_count = 0
    
    for doc in documents:
        result = collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"description_embedding": doc["description_embedding"]}}
        )
        updated_count += result.modified_count
    
    return updated_count

def add_description_embeddings(limit: int = None):
    """
    Main function to update description embeddings.
    
    Args:
        limit (int, optional): Maximum number of documents to process. Defaults to None.
    """
    # Step 1: Get documents without embeddings
    documents = get_documents_without_embeddings(limit)
    print(f"Found {len(documents)} documents to process")
    
    # Process documents in batches of 50
    batch_size = 50
    total_updated = 0
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch = documents[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} documents)")
        
        # Step 2: Generate embeddings for current batch
        print("Step 2: Generating embeddings...")
        batch_with_embeddings = generate_embeddings_for_documents(batch)
        
        # Step 3: Update documents for current batch
        print("Step 3: Updating documents...")
        updated_count = update_documents_with_embeddings(batch_with_embeddings)
        total_updated += updated_count
        print(f"Updated {updated_count} documents in this batch")
        print(f"Total documents updated so far: {total_updated}")
    
    print(f"\nAll batches processed. Total documents updated: {total_updated}")
    return

if __name__ == "__main__":
    add_description_embeddings(limit=None) 