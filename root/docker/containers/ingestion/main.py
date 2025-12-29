"""
Ingestion Service for Ivy AI Orchestration System

This service processes PDF documents and uploads them to Qdrant vector database using
ColQwen2.5 Omni embeddings.

Architecture: 
- Monitors /app/data/unprocessed/ for PDFs
- Converts PDF pages to images (150 DPI)
- Generates embeddings using ColQwen2.5 Omni
- Stores multivector embeddings in Qdrant
- Moves processed PDFs to /app/data/processed/
"""

import os
import glob
import uuid
import time
from typing import Tuple, List, Optional, Dict, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path, pdfinfo_from_path

from colpali_engine.models import ColQwen2_5Omni, ColQwen2_5OmniProcessor
from qdrant_client import QdrantClient, models


# Constants
QDRANT_COLLECTION_NAME = "embeddings_database"
EMBEDDING_DIMENSION = 128
PDF_DPI = 150
PAGES_PER_BATCH = 1  # Process one page at a time due to compute constraints
DATA_DIR = "/app/data"
UNPROCESSED_DIR = f"{DATA_DIR}/unprocessed"
PROCESSED_DIR = f"{DATA_DIR}/processed"

def embed_images(
    image_batch: List[Image.Image],
    model_processor: ColQwen2_5OmniProcessor,
    model: ColQwen2_5Omni
) -> List[List[float]]:
    """
    Generate embeddings for a batch of images using ColQwen2.5 Omni.
    
    Creates multivector embeddings preserving spatial structure from the image.
    
    Args:
        image_batch: List of PIL Image objects to embed
        model_processor: ColQwen2_5OmniProcessor for image preprocessing
        model: ColQwen2_5Omni model for generating embeddings
        
    Returns:
        List of multivector embeddings (one per image)
    """
    # Generate embeddings without gradient computation (inference only)
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device) 
        image_embeddings = model(**processed_images)

    original_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()

    return original_embeddings_batch

def get_pdf_info(pdf_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_path: Absolute path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata (including 'Pages' key),
        or None if extraction fails
    """
    try:
        info = pdfinfo_from_path(pdf_path)
        return info
    except Exception as e:
        print(f"Error extracting PDF metadata from {pdf_path}: {e}")
        return None

def main() -> None:
    """
    Main ingestion pipeline entry point.
    
    Workflow:
    1. Connect to Qdrant vector database
    2. Initialize or verify collection with multivector configuration
    3. Load ColQwen2.5 Omni model for embedding generation
    4. Process all PDFs in unprocessed directory
    5. Move completed PDFs to processed directory
    """
    print("Ingestion service started!")
    
    # Initialize Qdrant client using environment variable
    qdrant_url = os.getenv("QDRANT_URL")
    client = QdrantClient(url=qdrant_url)
    print(f"Connected to Qdrant at {qdrant_url}")
    
    # Ensure collection exists with proper multivector configuration
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")
    else:
        print(f"Creating collection '{QDRANT_COLLECTION_NAME}' with multivector config...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                # Original multivector embeddings
                # HNSW disabled (m=0) for speed since these are only used for reranking
                "original": models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW index
                )
            }
        )
    
    def upload_embeddings_to_qdrant(
        original_embeddings: np.ndarray,
        metadata_batch: List[Dict[str, Any]],
        collection_name: str
    ) -> None:
        """
        Upload a batch of embeddings with metadata to Qdrant.
        
        Uploads the original embeddings as named vectors within the same collection for retrieval.
        
        Args:
            original_embeddings: Full multivector embeddings
            metadata_batch: List of payload dictionaries (source, filename, page_number, etc.)
            collection_name: Target Qdrant collection name
        """
        try:
            client.upload_collection(
                collection_name=collection_name,
                vectors={
                    "original": original_embeddings
                },
                payload=metadata_batch,
                ids=[str(uuid.uuid4()) for _ in range(len(original_embeddings))]
            )
        except Exception as e:
            print(f"Error uploading to Qdrant: {e}")
    
    # Initialize ColQwen2.5 Omni model for embedding generation
    print("Loading ColQwen2.5 Omni model and processor...")
    colqwen_model = ColQwen2_5Omni.from_pretrained(
        "vidore/colqwen-omni-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Automatically distribute model across available devices
        low_cpu_mem_usage=True  # Minimize CPU memory during loading
    ).eval()

    colqwen_processor = ColQwen2_5OmniProcessor.from_pretrained("vidore/colqwen-omni-v0.1")
    
    # Discover PDFs to process from unprocessed directory
    while True: 
        source_label = "ingestion"  # Label for tracking document source
        pdf_files = glob.glob(os.path.join(UNPROCESSED_DIR, "*.pdf"))
        print(f"Found {len(pdf_files)} PDF file(s) to process")
        
        # Process each PDF file
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            print(f"\nProcessing {filename}...")
            
            # Extract PDF metadata
            pdf_info = get_pdf_info(pdf_path)
            total_pages = pdf_info.get('Pages', 0) if pdf_info else 0
            print(f"  Document has {total_pages} page(s)")
            
            # Process PDF in batches of pages
            for start_page_num in range(1, total_pages + 1, PAGES_PER_BATCH):
                end_page_num = min(start_page_num + PAGES_PER_BATCH - 1, total_pages)
                
                # Convert PDF pages to images
                page_images = convert_from_path(
                    pdf_path,
                    dpi=PDF_DPI,
                    first_page=start_page_num,
                    last_page=end_page_num,
                    fmt="JPEG"
                )
                print(f"  Processing page(s) {start_page_num}-{end_page_num}...")
                
                # Embed and upload each page
                for page_offset, page_image in enumerate(tqdm(
                    page_images, desc="  Generating embeddings"
                )):
                    current_page_num = start_page_num + page_offset
                    
                    try:
                        # Generate embedding for the page
                        original = embed_images(
                            [page_image],
                            colqwen_processor,
                            colqwen_model
                        )
                        
                        # Upload to Qdrant with metadata
                        upload_embeddings_to_qdrant(
                            np.asarray(original, dtype=np.float32),
                            [{
                                "source": source_label,
                                "filename": filename,
                                "page_number": current_page_num,
                                "doc_type": "pdf"
                            }],
                            QDRANT_COLLECTION_NAME
                        )
                        
                    except Exception as e:
                        print(f"  Error processing page {current_page_num}: {e}")
                        continue
            
            # Move successfully processed PDF to processed directory
            try:
                destination_path = os.path.join(PROCESSED_DIR, filename)
                os.rename(pdf_path, destination_path)
                print(f"  Moved {filename} to processed directory")
            except Exception as e:
                print(f"  Error moving {filename}: {e}")
                
        # Sleep before checking for new files
        print("Sleeping for 1 minute before checking for new files...")
        time.sleep(60)

if __name__ == "__main__":
    main()
