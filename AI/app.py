from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import time
print('Importing ImagePreprocessing.py...')
import ImagePreprocessing
print('Importing DeleteUser.py...')
import DeleteUser

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

DATASET_DIR = "C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return "In progress, go to /upload"

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'files[]' not in request.files or 'user_name' not in request.form:
            flash('No file part or user name provided')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        user_name = request.form['user_name']
        
        if len(files) != 5:
            flash('You must upload exactly 5 images.')
            return redirect(request.url)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                flash('Allowed image types are - jpg, jpeg')
                return redirect(request.url)

        # Run the script to store the images
        store_images(user_name)
        time.sleep(.2)  # to make sure all images are stored successfully
        ImagePreprocessing.main(user_name, 'Y')  # 'Y' = User is authenticated to enter

        flash('Images successfully uploaded and processed!')
        return redirect(url_for('upload_files'))

    return render_template('upload.html')

@app.route('/delete', methods=['GET', 'POST'])
def delete_user():
    if request.method == 'POST':
        user_name = request.form['user_name']
        if not user_name:
            flash('User name is required.')
            return redirect(request.url)

        try:
            DeleteUser.main(user_name)
            flash('Delete Successful!')
        except Exception as e:
            flash(f'Error: {e}')

        return redirect(url_for('delete_user'))

    return render_template('delete.html')

def store_images(user_name):
    # Determine the next folder name
    existing_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]

    if existing_folders:
        max_folder_num = max([int(f) for f in existing_folders])
        new_folder_num = max_folder_num + 1
    else:
        new_folder_num = 1

    new_folder_path = os.path.join(DATASET_DIR, str(new_folder_num))
    unfiltered_folder_path = os.path.join(new_folder_path, 'Unfiltered')

    # Create new folders
    os.makedirs(unfiltered_folder_path)

    # Move uploaded images to the new folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dest_path = os.path.join(unfiltered_folder_path, filename)
        os.rename(src_path, dest_path)
        print(f"Stored {filename} in {unfiltered_folder_path}")

    # You can use the user_name variable here if needed
    print(f"Images stored for user: {user_name}")


if __name__ == "__main__":
    app.run(debug=True, port=1234)
