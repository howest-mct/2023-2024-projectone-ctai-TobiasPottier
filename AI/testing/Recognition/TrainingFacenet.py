import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from keras_facenet import FaceNet
import cv2
import pickle

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

# Generate embeddings for each image and store them in a list
embeddings = []
for image_path in df['image_path']:
    embedding = preprocess_and_embed(image_path)
    embeddings.append(embedding)

# Convert the list of embeddings to a numpy array
embeddings = np.array(embeddings)

# Add embeddings to the DataFrame as a new column
df['embedding'] = list(embeddings)

# Prepare data for training
X = np.vstack(df['embedding'].values)
y = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train a classifier (e.g., SVM)
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X_train, y_train)

# Evaluate the classifier
accuracy = classifier.score(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save the classifier and label encoder
with open('facenet_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved.")
