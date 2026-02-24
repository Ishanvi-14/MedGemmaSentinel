import lancedb
import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict

class GuidelineVectorStore:
    """
    Handles local indexing of NCCN Guidelines using LanceDB.
    Meets Edge AI requirements by running entirely on-device.
    """
    def __init__(self, db_path: str = "./data/lancedb_sentinel"):
        self.db_path = db_path
        self.db = lancedb.connect(self.db_path)
        # Using a specialized medical or lightweight embedder
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.table_name = "nsclc_guidelines"

    def ingest_nccn_guidelines(self, pdf_folder: str):
        """Parses and indexes PDF guidelines into LanceDB."""
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_folder, file))
                pages = loader.load()
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)

        # Prepare data for LanceDB
        data = []
        for i, chunk in enumerate(all_chunks):
            vector = self.embeddings.embed_query(chunk.page_content)
            data.append({
                "id": i,
                "vector": vector,
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "NCCN Guidelines"),
                "page": chunk.metadata.get("page", 0)
            })

        self.db.create_table(self.table_name, data=data, mode="overwrite")

    def query_guidelines(self, query: str, top_k: int = 3) -> List[Dict]:
        """Performs semantic search to find relevant treatment protocols."""
        table = self.db.open_table(self.table_name)
        query_vector = self.embeddings.embed_query(query)
        
        # Simple search (No complex filtering per Rule 2)
        results = table.search(query_vector).limit(top_k).to_pandas()
        return results.to_dict('records')