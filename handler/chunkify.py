from typing import Dict, List, Optional

import tiktoken
from handler.utils import hash_int

from models.document import DocumentChunk, DocumentChunkMetadata, SingleDocument, SingleDocumentWithChunks


# Global variables
tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # The encoding scheme to use for tokenization

# Constants
CHUNK_SIZE = 200  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = 128  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text

def _chunkify(
        text: str,
        chunk_token_len: Optional[int]
) -> List[str]:
    if text is None or text.isspace():
        return []
    tokens = tokenizer.encode(text, disallowed_special=())
    chunks = []
    chunk_size = chunk_token_len or CHUNK_SIZE
    num_chunks = 0
    while tokens and num_chunks < MAX_NUM_CHUNKS:
        chunk = tokens[:chunk_size]
        chunk_text = tokenizer.decode(chunk)
        if chunk_text is None or chunk_text.isspace():
            tokens = tokens[len(chunk) :]
            continue
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind(";"),
            chunk_text.rfind("\n"),
        )
        if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
            chunk_text = chunk_text[: last_punctuation + 1]
        chunk_text_to_append = chunk_text.replace("\n", " ").strip()

        if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(chunk_text_to_append)

        tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())) :]
        num_chunks += 1
    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(remaining_text)
    return chunks    
    

def _add_chunks_to_doc(
        document: SingleDocument,
        chunk_token_len: Optional[int]
) -> Optional[SingleDocumentWithChunks]:
    if document.text is None or document.text.isspace():
        return None
    doc_id = document.doc_id
    doc_metadata = document.metadata
    chunk_metadata = DocumentChunkMetadata(doc_id=doc_id, doc_metadata=doc_metadata)
    chunk_texts = _chunkify(document.text, chunk_token_len)
    # chunk_id is defined by both the chunk text and the document it belongs to
    # note that chunk_id of a chunk in different versions of a document
    # won't change if the texts are the same
    chunk_ids = [hash_int(text + str(hash(document))) for text in chunk_texts]
    chunks = [DocumentChunk(chunk_id=chunk_id,
                            text=chunk_text,
                            metadata=chunk_metadata)
            for chunk_id, chunk_text 
            in zip(chunk_ids, chunk_texts)]
    return SingleDocumentWithChunks(**document.dict(), chunks=chunks)
    
    

def get_document_chunks(
        documents: List[SingleDocument],
        chunk_token_len: Optional[int]
) -> List[SingleDocumentWithChunks]:
    docs_with_chunks: List[SingleDocumentWithChunks] = []
    for doc in documents:
        doc_with_chunks = _add_chunks_to_doc(doc, chunk_token_len)
        if not doc_with_chunks:
            continue
        docs_with_chunks.append(doc_with_chunks)
    return docs_with_chunks
