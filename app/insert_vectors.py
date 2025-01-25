from datetime import datetime
import pandas as pd
import uuid
import psycopg2
from database.vector_store import VectorStore

# Define the uuid_from_time function
def uuid_from_time(timestamp):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp)))

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname="mydatabase", user="ritika02", password="ritika", host="localhost", port="5432"
)


# Initialize VectorStore with the database connection
vec = VectorStore(conn)

# Read the CSV file
df = pd.read_csv(r"C:\Users\Ritika\Downloads\pgvectorscale-rag-solution-main\pgvectorscale-rag-solution-main\data\faq_dataset.csv", sep=";")

print(df.head())

# Prepare data for insertion
def prepare_record(row):
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)  # Generate embedding using SentenceTransformer
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),  # Assuming uuid_from_time is correctly imported
            "metadata": {
                "category": row["category"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )

# Prepare the records DataFrame
records_df = df.apply(prepare_record, axis=1)
print("Prepared Records DataFrame:")
print(records_df)

# Create tables and insert data
print("Creating tables...")
vec.create_tables()  # Ensure this method exists in your VectorStore class if needed
print("Tables created successfully.")

print("Creating index...")
vec.create_index()  # Ensure this method exists in your VectorStore class if needed
print("Index created successfully.")

print("Upserting records into the database...")
vec.upsert(records_df)  # Upsert the DataFrame
print("Records inserted successfully.")

# Close the database connection
conn.close()
