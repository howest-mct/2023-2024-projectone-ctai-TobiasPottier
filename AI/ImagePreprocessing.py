#region Imports
print('Importing Libraries...')
import os
from PIL import Image
import cv2
import os
from ultralytics import YOLO
import numpy as np
import imgaug.augmenters as iaa
from glob import glob
import pandas as pd
from keras_facenet import FaceNet
import pickle
import mysql.connector
import shutil
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

#endregion
#region Preprocessing
print('Importing YOLO model...')
model = YOLO('./detectionModel3.pt')

def check_user_name_exists(user_name, cursor):
    # Check if the user name exists in the face_name table
    cursor.execute("SELECT 1 FROM face_name WHERE FaceName = %s", (user_name,))
    result = cursor.fetchone()

    if result:
        raise Exception(f"User name {user_name} already exists in the database.")

def move_uploaded_images(upload_folder, dataset_dir):
    # Determine the next folder name
    existing_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    if existing_folders:
        max_folder_num = max([int(f) for f in existing_folders])
        new_folder_num = max_folder_num + 1
    else:
        new_folder_num = 1

    new_folder_path = os.path.join(dataset_dir, str(new_folder_num))

    # Create new folders
    os.makedirs(new_folder_path)

    # Move uploaded images to the new folder
    for filename in os.listdir(upload_folder):
        src_path = os.path.join(upload_folder, filename)
        dest_path = os.path.join(new_folder_path, filename)
        os.rename(src_path, dest_path)
        print(f"Stored {filename} in {new_folder_path}")

    return new_folder_path

def detect_and_crop_faces(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Load the image
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Perform face detection
            results = model(image)

            # Load the image using OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Process the results and show only the detected faces
            index = 1
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Cut out the face from the frame
                    face = cv_image[y1:y2, x1:x2]

                    # Save the cropped face image
                    face_filename = f"{os.path.splitext(filename)[0]}_face_{index}.jpg"
                    face_path = os.path.join(output_folder, face_filename)
                    cv2.imwrite(face_path, face)

                    print(f"Saved cropped face to {face_path}")
                    index+=1


def augment_images_in_directory(image_dir):
    # Function to augment images
    def augment_images(image_paths, augmenters, output_dir):
        for image_path in image_paths:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Apply augmentations
            for augmenter in augmenters:
                augmented_image = augmenter(image=image)
                # Create a new file name
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                augmenter_name = augmenter.name.replace('(', '').replace(')', '')  # Clean up augmenter name
                new_image_path = os.path.join(output_dir, f"{name}_aug_{augmenter_name}{ext}")
                
                # Save augmented image
                cv2.imwrite(new_image_path, augmented_image)
                print(f"Saved augmented image: {new_image_path}")

    # Define the augmenters
    brightness_augmenter = iaa.Multiply((0.5, 1.5)).to_deterministic()  # Adjust brightness
    rotation_augmenter = iaa.Affine(rotate=(-25, 25)).to_deterministic()  # Rotate images
    noise_augmenter = iaa.AdditiveGaussianNoise(scale=(10, 60)).to_deterministic()  # Add Gaussian noise
    flip_augmenter = iaa.Fliplr(1.0).to_deterministic()  # Flip images horizontally
    black_box_augmenter = iaa.CoarseDropout(0.02, size_percent=0.5).to_deterministic()  # Add black box

    augmenters = [brightness_augmenter, rotation_augmenter, noise_augmenter, flip_augmenter, black_box_augmenter]

    # Get all image paths
    image_paths = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.jpeg"))

    # Augment images and save them
    augment_images(image_paths, augmenters, image_dir)


def embed_and_store_images(image_dir, cursor):
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

    # Function to insert data into MySQL
    def insert_data(embedding, label):
        binary_embedding = pickle.dumps(embedding)
        cursor.execute(
            "INSERT INTO face_data (embedding, label) VALUES (%s, %s)",
            (binary_embedding, label)
        )

    # Traverse through the directory and embed images
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            label = os.path.basename(image_dir)
            embedding = preprocess_and_embed(image_path)
            insert_data(embedding, label)

    print("Embeddings and labels have been successfully inserted into the database.")

def insert_auth_label(label_id, cursor):
    # Insert the LabelID into the auth table
    cursor.execute(
        "INSERT INTO auth (LabelID) VALUES (%s)",
        (label_id,)
    )

    print(f"LabelID {label_id} has been successfully inserted into the auth table.")

def insert_label_and_name(label_id, user_name, user_password, cursor):
    # Insert the LabelID and FaceName into the face_name table
    cursor.execute(
        "INSERT INTO face_name (LabelID, FaceName, FacePassword) VALUES (%s, %s, %s)",
        (label_id, user_name, user_password)
    )

    print(f"LabelID {label_id} and UserName {user_name} have been successfully inserted into the face_name table.")


def TrainAndManageClassifier(current_classifier_dir, backup_classifier_dir, cursor):
    # Ensure backup directory exists
    print('Backing Up Current Classifier...')
    if not os.path.exists(backup_classifier_dir):
        os.makedirs(backup_classifier_dir)

    # List all existing classifiers in the backup directory
    existing_classifiers = [f for f in os.listdir(backup_classifier_dir) if f.startswith("SVM_classifier")]
    
    # Determine the next index for the backup classifier
    if existing_classifiers:
        indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_classifiers]
        next_index = max(indices) + 1
    else:
        next_index = 1

    # Backup the current classifier
    current_classifier_path = os.path.join(current_classifier_dir, "SVM_classifier.pkl")
    backup_classifier_path = os.path.join(backup_classifier_dir, f"SVM_classifier_{next_index}.pkl")
    shutil.move(current_classifier_path, backup_classifier_path)
    print(f"Current classifier backed up to: {backup_classifier_path}")

    # Train the new classifier
    print('-TRAINING MODEL-')
    print('Collecting Data...')

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

    # Prepare data for training
    X = np.vstack(df['embedding'].values)
    y = df['label'].values


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

    train_indices, val_indices = custom_train_val_split(y=y, val_size=0.1)

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    print('Training SVM model...')
    # Train a classifier SVM
    # finding best SVM model using Cross Validation Gridsearch

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

    print('Best accuracy : ', grid_search.best_score_)
    print('Best parameters :', grid_search.best_params_)

    SVMmodel = SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'], probability=True)
    if 'gamma' in grid_search.best_params_:
        SVMmodel = SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'], gamma=grid_search.best_params_['gamma'], probability=True)

    SVMmodel.fit(X_train, y_train)

    y_pred = SVMmodel.predict(X_val)
    print(classification_report(y_val, y_pred))

    cf = confusion_matrix(y_val, y_pred)
    print(cf)

    accuracy = accuracy_score(y_val, y_pred) * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save the new classifier in the current classifier directory
    with open(current_classifier_path, 'wb') as f:
        pickle.dump(SVMmodel, f)
    print(f"New classifier saved at: {current_classifier_path}")



def create_flag_file(flag_file):
    # Create the flag file to signal the real-time script
    with open(flag_file, 'w') as f:
        f.write("Reload classifier")



#endregion 

#region FindFolder

def check_and_preprocess(folder_path, current_classifier_dir, backup_classifier_dir, user_name, flag_file, user_password, cursor):
    """Check for READY.txt file in the specified folder and create it if not present."""
    ready_file_path = os.path.join(folder_path, 'READY.txt')

    if not os.path.exists(ready_file_path):
        print("READY.txt does not exist in the folder. OK")
        print('Augmenting Pictures...')
        augment_images_in_directory(folder_path)
        time.sleep(1) # wait time to make sure all images are made before embedding
        print('Embedding Pictures and Storing in SQL Database...')
        embed_and_store_images(folder_path, cursor)
        print('Entering LabelID in AUTH SQL Database...')
        insert_auth_label(os.path.basename(folder_path), cursor)
        print('Entering Label, UserName and password in face_name SQL Database')
        insert_label_and_name(os.path.basename(folder_path), user_name, user_password, cursor)
        print('Training Classifier...')
        TrainAndManageClassifier(current_classifier_dir=current_classifier_dir, backup_classifier_dir=backup_classifier_dir, cursor=cursor)
        print('Creating Flag File... (./flag/)')
        create_flag_file(flag_file)
        print('Creating Ready.txt in Pictures Folder...')
        with open(ready_file_path, 'w') as f:
            f.write("This is the READY.txt file.")
        print("READY.txt file has been created.")
        print('ALL PROCESSES COMPLETE!')
        return
    else:
        print("READY.txt already exists in the folder.")
        return



def main(user_name, user_password, images_processed_event):
    dataset_dir = "./dataset"
    current_classifier_dir = "./"
    backup_classifier_dir = "./BackupModels"
    flag_file = "./flag/reload_flag.txt"
    upload_folder = './captures'

    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='face_recognition'
    )
    cursor = conn.cursor()

    try:
        check_user_name_exists(user_name, cursor)
        new_folder_path = move_uploaded_images(upload_folder, dataset_dir)
        print(f"New folder created: {new_folder_path}")
        check_and_preprocess(new_folder_path, current_classifier_dir, backup_classifier_dir, user_name, flag_file, user_password, cursor)
        conn.commit()
        images_processed_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

#endregion

if __name__ == "__main__":
    print('Script ran from __main__, no changes made')
