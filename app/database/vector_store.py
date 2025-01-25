import logging
import time
import psycopg2
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
from uuid import uuid5
import uuid
from datetime import datetime

class VectorStore:
    def __init__(self, connection):
        """Initialize the VectorStore with a local embedding model and database connection."""
        self.connection = connection
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_tables(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS vectors (
            id UUID PRIMARY KEY,
            metadata JSONB,
            contents TEXT,
            embedding FLOAT8[]
        );
        """
        cursor = self.connection.cursor()
        cursor.execute(create_table_query)
        self.connection.commit()
        cursor.close()
        logging.info("Tables created successfully.")

    def create_index(self):
        """Create an index for efficient searching."""
        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_vectors_embedding ON vectors USING ivfflat (embedding vector_l2_ops);
        """
        cursor = self.connection.cursor()
        cursor.execute(create_index_query)
        self.connection.commit()
        cursor.close()
        logging.info("Index created successfully.")

    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = self.embedding_model.encode(text).tolist()
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def upsert(self, df: pd.DataFrame) -> None:
        logging.info(f"Simulating upserting {len(df)} records into the database.")
        cursor = self.connection.cursor()
        for _, row in df.iterrows():
            insert_query = """
            INSERT INTO vectors (id, metadata, contents, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE 
            SET metadata = EXCLUDED.metadata, contents = EXCLUDED.contents, embedding = EXCLUDED.embedding;
            """
            cursor.execute(insert_query, (
                row["id"], row["metadata"], row["contents"], row["embedding"]
            ))
        self.connection.commit()
        cursor.close()
        logging.info(f"Upserted {len(df)} records into the database.")

    def search(self, query_text: str, limit: int = 5) -> List[dict]:
        query_embedding = self.get_embedding(query_text)
        return [{"id": "example_id", "similarity": 0.95, "metadata": {"example": "data"}}]

# Ensure proper connection initialization in main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(
        dbname="mydatabase", user="ritika02", password="ritika", host="localhost", port="5432"
    )
    vs = VectorStore(conn)
    vs.create_tables()
    example_text = "This is an example text to generate embeddings."
    embedding = vs.get_embedding(example_text)
    print("Generated Embedding:", embedding[:5])
    conn.close()
