"""
Ingestion Service for Ivy AI Orchestration System

This service processes PDF documents and uploads them to Qdrant vector database using
ColQwen2 embeddings. It implements a three-vector strategy (original, row-pooled, 
column-pooled) for optimal retrieval performance.

Architecture: 
- Monitors /app/data/unprocessed/ for PDFs
- Converts PDF pages to images (150 DPI)
- Generates multivector embeddings using ColQwen2
- Stores in Qdrant with spatial pooling variants
- Moves processed PDFs to /app/data/processed/
"""

import os
import glob
import uuid
from typing import Tuple, List, Optional, Dict, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path, pdfinfo_from_path

from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient, models


# Constants
QDRANT_COLLECTION_NAME = "RAG_ColQwen2"
EMBEDDING_DIMENSION = 128
PDF_DPI = 150
PAGES_PER_BATCH = 1  # Process one page at a time due to compute constraints
DATA_DIR = "/app/data"
UNPROCESSED_DIR = f"{DATA_DIR}/unprocessed"
PROCESSED_DIR = f"{DATA_DIR}/processed"


def get_patches(
    image_size: Tuple[int, int],
    model_processor: ColQwen2Processor,
    model: ColQwen2,
    model_name: str
) -> Tuple[Optional[int], Optional[int]]:
    """
    Calculate the number of patches (grid dimensions) for an image.
    
    ColPali models use a fixed 32x32 grid, while ColQwen2 calculates dimensions
    dynamically based on image size and spatial_merge_size parameter.
    
    Args:
        image_size: (width, height) tuple of the image
        model_processor: Processor instance for the model
        model: The vision model instance
        model_name: Either "colPali" or "colQwen"
        
    Returns:
        (rows, columns) tuple representing the patch grid dimensions,
        or (None, None) if model_name is unrecognized
    """
    if model_name == "colPali":
        return model_processor.get_n_patches(
            image_size,
            patch_size=model.patch_size
        )
    elif model_name == "colQwen":
        # ColQwen2 processor calculates patch_size from model internally
        return model_processor.get_n_patches(
            image_size,
            spatial_merge_size=model.spatial_merge_size
        )
    return None, None


def embed_and_mean_pool_batch(
    image_batch: List[Image.Image],
    model_processor: ColQwen2Processor,
    model: ColQwen2,
    model_name: str
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Generate embeddings for a batch of images with spatial pooling variants.
    
    Creates three representations for each image:
    1. Original multivector embeddings (~700 vectors/page, preserves spatial structure)
    2. Row-wise mean pooled embeddings (HNSW-optimized for fast retrieval)
    3. Column-wise mean pooled embeddings (HNSW-optimized for fast retrieval)
    
    Args:
        image_batch: List of PIL Image objects to embed
        model_processor: ColQwen2 processor for image preprocessing
        model: ColQwen2 model for generating embeddings
        model_name: Model identifier ("colPali" or "colQwen")
        
    Returns:
        Tuple of (original_embeddings, row_pooled_embeddings, column_pooled_embeddings)
        Each is a list of embedding vectors (one per image)
    """
    # Generate embeddings without gradient computation (inference only)
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device) 
        image_embeddings = model(**processed_images)

    original_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()
    
    # Prepare lists for spatially pooled embeddings
    row_pooled_batch = []
    column_pooled_batch = []
    
    
    # Process each image in the batch
    for image_embedding, tokenized_image, image in zip(
        image_embeddings,
        processed_images.input_ids,
        image_batch
    ):
        num_rows, num_columns = get_patches(
            image.size, model_processor, model, model_name
        )
        
        # Extract only the image patch tokens (excluding special tokens)
        image_tokens_mask = (tokenized_image == model_processor.image_token_id)
        
        # Reshape flat embedding into spatial grid: (rows, columns, embedding_dim)
        image_tokens = image_embedding[image_tokens_mask].view(
            num_rows, num_columns, model.dim
        )
        
        # Mean pool across each dimension
        row_pooled = torch.mean(image_tokens, dim=0)  # Pool across rows -> (columns, dim)
        column_pooled = torch.mean(image_tokens, dim=1)  # Pool across columns -> (rows, dim)

        # Locate special tokens (prefix/postfix) in the tokenized sequence
        image_token_indices = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_indices[0].cpu().item()
        last_image_token_idx = image_token_indices[-1].cpu().item()
        
        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1:]

        # Reconstruct embeddings with special tokens preserved
        # Format: [prefix_tokens] + [pooled_embeddings] + [postfix_tokens]
        row_pooled_with_special = torch.cat(
            (prefix_tokens, row_pooled, postfix_tokens), dim=0
        ).cpu().float().numpy().tolist()
        
        column_pooled_with_special = torch.cat(
            (prefix_tokens, column_pooled, postfix_tokens), dim=0
        ).cpu().float().numpy().tolist()
        
        row_pooled_batch.append(row_pooled_with_special)
        column_pooled_batch.append(column_pooled_with_special)

    return original_embeddings_batch, row_pooled_batch, column_pooled_batch

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
    3. Load ColQwen2 model for embedding generation
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
                # Original multivector embeddings (~700 vectors/page)
                # HNSW disabled (m=0) for speed since these are only used for reranking
                "original": models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW index
                ),
                # Column-pooled embeddings for fast prefetch stage
                # HNSW enabled (default) for efficient similarity search
                "mean_pooling_columns": models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                ),
                # Row-pooled embeddings for fast prefetch stage
                # HNSW enabled (default) for efficient similarity search
                "mean_pooling_rows": models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            }
        )
    
    def upload_embeddings_to_qdrant(
        original_embeddings: np.ndarray,
        row_pooled_embeddings: np.ndarray,
        column_pooled_embeddings: np.ndarray,
        metadata_batch: List[Dict[str, Any]],
        collection_name: str
    ) -> None:
        """
        Upload a batch of embeddings with metadata to Qdrant.
        
        Uploads all three embedding variants (original, row-pooled, column-pooled)
        as named vectors within the same collection for two-stage retrieval.
        
        Args:
            original_embeddings: Full multivector embeddings
            row_pooled_embeddings: Row-wise mean pooled embeddings
            column_pooled_embeddings: Column-wise mean pooled embeddings
            metadata_batch: List of payload dictionaries (source, filename, page_number, etc.)
            collection_name: Target Qdrant collection name
        """
        try:
            client.upload_collection(
                collection_name=collection_name,
                vectors={
                    "mean_pooling_columns": column_pooled_embeddings,
                    "original": original_embeddings,
                    "mean_pooling_rows": row_pooled_embeddings
                },
                payload=metadata_batch,
                ids=[str(uuid.uuid4()) for _ in range(len(original_embeddings))]
            )
        except Exception as e:
            print(f"Error uploading to Qdrant: {e}")
    
    # Discover PDFs to process from unprocessed directory
    source_label = "ingestion"  # Label for tracking document source
    pdf_files = glob.glob(os.path.join(UNPROCESSED_DIR, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    if len(pdf_files) == 0:
        print("No PDFs to process. Exiting.")
    
    # Initialize ColQwen2 model for embedding generation
    print("Loading ColQwen2 model and processor...")
    colqwen_model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Options: "cuda:0" (GPU), "cpu", "mps" (Apple Silicon)
    ).eval()

    colqwen_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
    
    # Track global page index across all documents
    global_page_index = 0
    
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
                    # Generate three embedding variants for the page
                    original, row_pooled, column_pooled = embed_and_mean_pool_batch(
                        [page_image],
                        colqwen_processor,
                        colqwen_model,
                        "colQwen"
                    )
                    
                    # Upload to Qdrant with metadata
                    upload_embeddings_to_qdrant(
                        np.asarray(original, dtype=np.float32),
                        np.asarray(row_pooled, dtype=np.float32),
                        np.asarray(column_pooled, dtype=np.float32),
                        [{
                            "source": source_label,
                            "filename": filename,
                            "page_number": current_page_num,
                            "index": global_page_index
                        }],
                        QDRANT_COLLECTION_NAME
                    )
                    global_page_index += 1
                    
                except Exception as e:
                    print(f"  Error processing page {current_page_num}: {e}")
                    continue
            
            # Free memory from processed batch
            del page_images
        
        # Move successfully processed PDF to processed directory
        try:
            destination_path = os.path.join(PROCESSED_DIR, filename)
            os.rename(pdf_path, destination_path)
            print(f"  Moved {filename} to processed directory")
        except Exception as e:
            print(f"  Error moving {filename}: {e}")
    
    print("\nIngestion pipeline completed successfully")
    
    ## Test query
    query = "What does DRD mean?"
    processed_queries = colqwen_processor.process_queries([query]).to(colqwen_model.device)
    query_embedding = colqwen_model(**processed_queries)[0].cpu().float().numpy().tolist()
    
    # Final amount of results to return
    search_limit = 10
    # Amount of results to prefetch for reranking
    prefetch_limit = 100

    response = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        prefetch=[
            models.Prefetch(
                query=query_embedding,
                limit=prefetch_limit,
                using="mean_pooling_columns"
            ),
            models.Prefetch(
                query=query_embedding,
                limit=prefetch_limit,
                using="mean_pooling_rows"
            ),
        ],
        limit=search_limit,
        with_payload=True,
        using="original"
    )
    
    # Print result:
    print("Test Query Results:")
    print(f"Query: {query}")
    for point in response.points:
        print(f"Score: {point.score}")
        print(f"Payload: {point.payload}")

if __name__ == "__main__":
    main()
