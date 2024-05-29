import mysql.connector
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shutil

def get_label_id_by_name(name, cursor):
    # Execute the SQL query to find the LabelID by name
    cursor.execute("SELECT LabelID FROM face_name WHERE FaceName = %s", (name,))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    else:
        return None
    
def verify_user_password(name, password, cursor):
    # Execute the SQL query to find the user by name and password
    cursor.execute("SELECT LabelID FROM face_name WHERE FaceName = %s AND FacePassword = %s", (name, password))
    result = cursor.fetchone()

    if not result:
        raise Exception(f"Incorrect password for user {name}")
    
def delete_row_face_name_by_label_id(label_id, cursor):
    # Execute the SQL query to delete the row by LabelID
    cursor.execute("DELETE FROM face_name WHERE LabelID = %s", (label_id,))

def delete_row_auth_by_label_id(label_id, cursor):
    # Execute the SQL query to delete the row by LabelID
    cursor.execute("DELETE FROM auth WHERE LabelID = %s", (label_id,))

def delete_row_face_data_by_label_id(label_id, cursor):
    # Execute the SQL query to delete the row by LabelID
    cursor.execute("DELETE FROM face_data WHERE Label = %s", (label_id,))

def delete_labels_csv_rows(csv_path, label_id):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Filter the DataFrame to remove rows with the given label_id
    df_filtered = df[df['label'] != label_id]
    
    # Save the updated DataFrame back to the CSV file
    df_filtered.to_csv(csv_path, index=False)
    print(f"Rows with label {label_id} have been deleted from the CSV file.")

def TrainAndReplaceModel(current_classifier_dir, backup_classifier_dir):
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

    # Here I would add code to send a signal to the real-time script to reload its model
    # For example, I might set a flag or send a message over a network, etc.
    # This part is left for later implementation.
    pass

def delete_files_in_label_folder(images_user_dataset_dir, label_id):
    # Construct the path to the folder with the given label_id
    target_folder_path = os.path.join(images_user_dataset_dir, str(label_id))
    
    if os.path.isdir(target_folder_path):
        # Delete all .jpg files and the READY.txt file in the specified folder
        for filename in os.listdir(target_folder_path):
            file_path = os.path.join(target_folder_path, filename)
            try:
                if filename.endswith('.jpg') or filename == 'READY.txt':
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # Check and delete the 'Unfiltered' folder if it exists
        unfiltered_folder_path = os.path.join(target_folder_path, 'Unfiltered')
        if os.path.isdir(unfiltered_folder_path):
            try:
                for filename in os.listdir(unfiltered_folder_path):
                    file_path = os.path.join(unfiltered_folder_path, filename)
                    os.remove(file_path)
                os.rmdir(unfiltered_folder_path)
                print(f"Deleted folder: {unfiltered_folder_path}")
            except Exception as e:
                print(f"Failed to delete folder {unfiltered_folder_path}. Reason: {e}")

        print('Creating Death File...')
        create_death_file(os.path.join(target_folder_path, "Death.txt"))
    else:
        print(f"Folder with label ID {label_id} not found at {images_user_dataset_dir}")

def create_death_file(death_file):
    with open(death_file, 'w') as f:
        f.write(" -- This LabelID has deleted all their records")

def create_flag_file(flag_file):
    # Create the flag file to signal the real-time script
    with open(flag_file, 'w') as f:
        f.write("Reload classifier")

def main(user_name, user_password):
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='face_recognition'
    )
    cursor = conn.cursor()
    labels_csv_dir = "C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubLabels/labels.csv"
    users_images_dataset_dir = "C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset"
    flag_file = "./flag/reload_flag.txt"
    try:
        print('Getting LabelID...')
        label_id = get_label_id_by_name(user_name, cursor)
        if label_id is not None:
            print(f"LabelID ({user_name}) = {label_id}")
            print('Verifying password...')
            try:
                verify_user_password(user_name, user_password, cursor)
            except Exception as ex:
                raise ex
            print('Deleting face_name...')
            delete_row_face_name_by_label_id(label_id, cursor)
            print('Deleting auth...')
            delete_row_auth_by_label_id(label_id, cursor)
            print('Deleting face_data...')
            delete_row_face_data_by_label_id(label_id, cursor)
            print('Deleting annotations in labels.csv...')
            delete_labels_csv_rows(labels_csv_dir, label_id)
            conn.commit()  # Commit all changes once at the end
            print('Deleting Images...')
            delete_files_in_label_folder(users_images_dataset_dir, label_id)
            print('Training Classifier...')
            current_classifier_dir = "./"
            backup_classifier_dir = "./BackupModels"
            TrainAndReplaceModel(current_classifier_dir, backup_classifier_dir)
            print('Creating Flag File... (./flag/)')
            create_flag_file(flag_file)
            print('All processes complete!')
        else:
            print(f"Name {user_name} not found in the database.")
            print('Delete CANCELLED')
            raise Exception(f"Name {user_name} not found in the database.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        conn.rollback()  # Rollback in case of error
    finally:
        cursor.close()
        conn.close() 

if __name__ == "__main__":
    # name = 'Tomas'
    # userInput = input(f'Are you sure you want to delete: {name} ?(Y/N): ')
    # if userInput.upper() != 'Y':
    #     print('Delete CANCELLED')
    #     exit()
    # userInput2 = input(f'Are you REALLY sure you want to delete FROM ALL DATABASES: {name} ?(Y/N): ')
    # if userInput2.upper() != 'Y':
    #     print('Delete CANCELLED')
    #     exit()
    # main(name)
    print('Script ran from __main__, no chages made')
