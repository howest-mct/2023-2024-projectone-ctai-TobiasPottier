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

def delete_uploaded_images(upload_folder):
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.route('/')
def index():
    return "In progress, go to /upload"

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'files[]' not in request.files or 'user_name' not in request.form or 'password' not in request.form:
            flash('No file part, user name, or password provided')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        user_name = request.form['user_name']
        user_password = request.form['password']
        
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

        try:
            # Script to store, augment, annotate, embed, clasify, retrain models and make flag file
            ImagePreprocessing.main(user_name, user_password, 'Y')  # 'Y' = User is authenticated to enter
            flash('Images successfully uploaded and processed!')
        except Exception as e:
            time.sleep(.2) # to makes sure all images are uploaded first before deleting them
            delete_uploaded_images(app.config['UPLOAD_FOLDER'])
            flash(f'Error: {e}')

        return redirect(url_for('upload_files'))

    return render_template('upload.html')

@app.route('/delete', methods=['GET', 'POST'])
def delete_user():
    if request.method == 'POST':
        user_name = request.form['user_name']
        user_password = request.form['password']
        if not user_name:
            flash('User name is required.')
            return redirect(request.url)

        try:
            DeleteUser.main(user_name, user_password)
            flash('Delete Successful!')
        except Exception as e:
            flash(f'Error: {e}')

        return redirect(url_for('delete_user'))

    return render_template('delete.html')


if __name__ == "__main__":
    app.run(debug=True, port=1234)
