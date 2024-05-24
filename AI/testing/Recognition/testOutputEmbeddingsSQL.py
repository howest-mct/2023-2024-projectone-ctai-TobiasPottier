import mysql.connector
import pickle
import numpy as np

# Connect to MySQL database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='face_recognition'
)
cursor = conn.cursor()

# Function to retrieve data from MySQL
def retrieve_data():
    cursor.execute("SELECT embedding, label FROM face_data")
    results = cursor.fetchall()
    embeddings = []
    labels = []
    for (binary_embedding, label) in results:
        embedding = pickle.loads(binary_embedding)
        embeddings.append(embedding)
        labels.append(label)
    return np.array(embeddings), np.array(labels)

# Retrieve data
embeddings, labels = retrieve_data()

for embedding, label in zip(embeddings, labels):
    print(f"Label: {label}")
    print(f"Embedding: {embedding}\n")

# Close the database connection
cursor.close()
conn.close()

print("Embeddings and labels have been successfully retrieved from the database.")
