import mysql.connector
import numpy as np
from langchain.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class MySQLVectorStore(VectorStore):
    def __init__(self, host, port, database, user, password, embedding, table_name="vector_store"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.embedding = embedding
        self.table_name = table_name

        # Create connection
        self.conn = mysql.connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        self.cursor = self.conn.cursor()

        # Create table if not exists
        self._create_table()

    def _create_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            document TEXT,
            embedding BLOB,
            source_id VARCHAR(255)
        )
        """
        self.cursor.execute(query)
        self.conn.commit()

    def from_texts(self, texts, metadatas=None, ids=None):
        """Convert texts to documents and add them to the vector store."""
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas or [])]
        self.add_documents(documents)

    def add_documents(self, documents):
        for doc in documents:
            embedding = self.embedding.embed_documents([doc.page_content])[0]  # Get embedding for document content

            embedding_blob = self._convert_to_blob(embedding)  # Convert to BLOB for MySQL
            query = f"INSERT INTO {self.table_name} (document, embedding, source_id) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (doc.page_content, embedding_blob, doc.metadata.get("source", "")))
        self.conn.commit()



    def similarity_search(self, query, k=3):
        """Perform a similarity search based on the query."""
        query_embedding = self.embedding.embed_query(query)  # Get query embedding using embed_query
        query_blob = self._convert_to_blob(query_embedding)

        # Perform a naive search (example: retrieve top_k documents)
        self.cursor.execute(f"""
        SELECT * FROM {self.table_name} ORDER BY id DESC LIMIT {k}
        """)
        rows = self.cursor.fetchall()

        # Return the documents (document text in the second column)
        return [row[1] for row in rows]  # Change from row[0] to row[1]

    
    def _convert_to_blob(self, vector):
        """Convert the numpy vector to a binary format (BLOB)."""
        return np.array(vector).tobytes()

# Initialize OpenAI embeddings
embedding = OpenAIEmbeddings()

# Initialize the custom MySQL vector store
vectorstore = MySQLVectorStore(
    host='localhost',
    port=3306,
    database='langchain_indexing',
    user='root',
    password='root',  # No password for root in your case
    embedding=embedding
)

# Create some test documents
doc1 = Document(page_content="paul paul paul", metadata={"source": r"C:\Users\Admin\Desktop\10-20-2024\data\paul_graham_essay.txt"})
doc2 = Document(page_content="state state", metadata={"source": r"C:\Users\Admin\Desktop\10-20-2024\data\state_of_the_union.txt"})

# Index documents
vectorstore.add_documents([doc1, doc2])

# Search for documents similar to "doggy"
results = vectorstore.similarity_search("state", k=3)

# Display results
for result in results:
    print(result)
