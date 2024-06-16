import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from keras_facenet import FaceNet
import cv2
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mysql.connector

print('Embedding Pictures...')
userInput = input('Are u sure u want to TRUNCATE the entire database and enter new data?: ')
if userInput.upper() == 'n':
    exit()
    
# Load the CSV file
df = pd.read_csv('C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubLabels/labels.csv')

# Load the FaceNet model
embedder = FaceNet()

# Function to preprocess and embed images
def preprocess_and_embed(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    embeddings = embedder.embeddings(img)
    return embeddings[0]

# Connect to MySQL database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='face_recognition'
)
cursor = conn.cursor()

# Function to truncate the table
def truncate_table():
    cursor.execute("TRUNCATE TABLE face_data")
    conn.commit()

# Function to insert data into MySQL
def insert_data(embedding, label):
    binary_embedding = pickle.dumps(embedding)
    cursor.execute(
        "INSERT INTO face_data (embedding, label) VALUES (%s, %s)",
        (binary_embedding, label)
    )
    conn.commit()

# Truncate the table before inserting new data
truncate_table()

# Generate embeddings for each image and insert them into the database
for index, row in df.iterrows():
    image_path = row['image_path']
    label = row['label']
    embedding = preprocess_and_embed(image_path)
    insert_data(embedding, label)

# Close the database connection
cursor.close()
conn.close()

print("Embeddings and labels have been successfully inserted into the database.")
