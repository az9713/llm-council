"""Embedding generation for semantic similarity search.

This module generates text embeddings for semantic caching.
Uses OpenAI's text-embedding model via OpenRouter API.
Falls back to simple hash-based approach if embedding fails.
"""

import httpx
import hashlib
import math
from typing import List, Optional, Dict, Any
from .config import OPENROUTER_API_KEY

# OpenRouter endpoint for embeddings
EMBEDDINGS_API_URL = "https://openrouter.ai/api/v1/embeddings"

# Default embedding model (OpenAI's text-embedding-3-small is fast and cheap)
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Embedding dimension for the default model
EMBEDDING_DIMENSION = 1536

# Fallback dimension for hash-based embeddings
HASH_EMBEDDING_DIMENSION = 256


async def get_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL
) -> Optional[List[float]]:
    """
    Generate embedding vector for text using OpenRouter embeddings API.

    Args:
        text: The text to embed
        model: Embedding model to use

    Returns:
        List of floats representing the embedding vector,
        or None if the request failed.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": text,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                EMBEDDINGS_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            # Extract embedding from response
            if 'data' in data and len(data['data']) > 0:
                embedding = data['data'][0].get('embedding')
                if embedding:
                    return embedding

            return None

    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def get_hash_embedding(text: str, dimension: int = HASH_EMBEDDING_DIMENSION) -> List[float]:
    """
    Generate a deterministic hash-based embedding for text.

    This is a fallback when API embeddings are unavailable.
    Uses multiple hash functions to create a pseudo-embedding.
    Not as semantically meaningful as real embeddings, but works
    for exact and near-exact matches.

    Args:
        text: The text to embed
        dimension: Dimension of the output vector

    Returns:
        List of floats representing the hash-based embedding
    """
    # Normalize text
    normalized = text.lower().strip()

    # Generate multiple hashes for different "dimensions"
    embedding = []
    for i in range(dimension):
        # Create a unique seed for each dimension
        seed = f"{i}:{normalized}"
        hash_bytes = hashlib.sha256(seed.encode()).digest()

        # Convert first 8 bytes to a float between -1 and 1
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big', signed=True)
        normalized_value = hash_int / (2**63)  # Normalize to [-1, 1]
        embedding.append(normalized_value)

    # Normalize the vector to unit length
    magnitude = math.sqrt(sum(x*x for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


async def get_embedding_with_fallback(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    use_api: bool = True
) -> Dict[str, Any]:
    """
    Get embedding for text, with fallback to hash-based approach.

    Args:
        text: The text to embed
        model: Embedding model to use
        use_api: Whether to try API embeddings first

    Returns:
        Dict containing:
        - embedding: List of floats
        - method: 'api' or 'hash'
        - model: Model used (or 'hash' for fallback)
        - dimension: Dimension of the embedding
    """
    if use_api:
        embedding = await get_embedding(text, model)
        if embedding:
            return {
                'embedding': embedding,
                'method': 'api',
                'model': model,
                'dimension': len(embedding)
            }

    # Fallback to hash-based embedding
    embedding = get_hash_embedding(text)
    return {
        'embedding': embedding,
        'method': 'hash',
        'model': 'hash',
        'dimension': len(embedding)
    }


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
        (1 = identical, 0 = orthogonal, -1 = opposite)
    """
    if len(vec1) != len(vec2):
        # Different dimensions - incompatible embeddings
        return 0.0

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate magnitudes
    mag1 = math.sqrt(sum(x * x for x in vec1))
    mag2 = math.sqrt(sum(x * x for x in vec2))

    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def find_most_similar(
    query_embedding: List[float],
    embeddings: List[Dict[str, Any]],
    threshold: float = 0.85
) -> Optional[Dict[str, Any]]:
    """
    Find the most similar embedding above threshold.

    Args:
        query_embedding: The embedding to search for
        embeddings: List of dicts with 'embedding' and 'data' keys
        threshold: Minimum similarity score to consider a match

    Returns:
        The most similar entry's data if similarity >= threshold,
        None otherwise
    """
    best_match = None
    best_score = threshold  # Start at threshold - only return if above

    for entry in embeddings:
        stored_embedding = entry.get('embedding', [])
        similarity = cosine_similarity(query_embedding, stored_embedding)

        if similarity > best_score:
            best_score = similarity
            best_match = {
                'data': entry.get('data'),
                'similarity': similarity,
                'query': entry.get('query', ''),
            }

    return best_match


def get_embedding_config() -> Dict[str, Any]:
    """
    Get embedding configuration for API endpoint.

    Returns:
        Dict with embedding configuration details
    """
    return {
        'default_model': DEFAULT_EMBEDDING_MODEL,
        'embedding_dimension': EMBEDDING_DIMENSION,
        'hash_dimension': HASH_EMBEDDING_DIMENSION,
        'api_url': EMBEDDINGS_API_URL,
        'description': 'Semantic embedding generation for response caching'
    }
