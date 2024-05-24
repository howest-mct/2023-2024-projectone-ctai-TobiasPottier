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

print('Collecting Data...')
# connect to mysql database
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
    return pd.DataFrame({'embedding': embeddings, 'label': labels})

# Retrieve data into a DataFrame
df = retrieve_data()
print('Data Succesfully Pulled From MySQL')

# Close the database connection
cursor.close()
conn.close()


# Prepare data for training
X = np.vstack(df['embedding'].values)
y = df['label'].values


# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

def custom_train_val_split(y, val_size=0.1):
    train_indices = []
    val_indices = []
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        split_point = int(len(label_indices) * (1 - val_size))
        train_indices.extend(label_indices[:split_point])
        val_indices.extend(label_indices[split_point:])
    return np.array(train_indices), np.array(val_indices)

train_indices, val_indices = custom_train_val_split(y=y_encoded, val_size=0.1)

X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y_encoded[train_indices], y_encoded[val_indices]

print('Training SVM model...')
# Train a classifier SVM
# finding best SVM model using Cross Validation Gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

model = SVC()
paramaters = [ 
        {'kernel': ['linear'], 'C': np.linspace(0.01,10000,10)}, # 10 models
        {'kernel': ['rbf'], 'C': np.linspace(0.01,10000,10), 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2]}, #10x5 = 50 models
        {'kernel': ['poly'], 'C':np.linspace(0.01,10000,10)} ] # 10 models
grid_search = GridSearchCV(estimator = model, 
                           param_grid = paramaters,
                           scoring = 'accuracy',
                           cv = 2, # k=5
                           n_jobs = -1,
                           verbose = 1)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_ 
best_parameters = grid_search.best_params_  

print('Best accuracy : ', grid_search.best_score_)
print('Best parameters :', grid_search.best_params_  )

SVMmodel = SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'], probability=True)
SVMmodel.fit(X_train, y_train)

y_pred = SVMmodel.predict(X_val)
print(classification_report(y_val, y_pred))

cf = confusion_matrix(y_val, y_pred)
print(cf)

accuracy = accuracy_score(y_val, y_pred) * 100
print(f"Validation Accuracy: {accuracy:.2f}%")

# Save the classifier and label encoder
with open('facenet_classifier.pkl', 'wb') as f:
    pickle.dump(SVMmodel, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved.")