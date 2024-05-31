from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import time
import threading
print('Importing ImagePreprocessing.py...')
import ImagePreprocessing
print('Importing DeleteUser.py...')
import DeleteUser
print('Importing TakeCameraPictures.py')
import TakeCameraPictures
print('Importing FullFaceRegnition.py')
import FullFaceRecognitionBLE

picture_index = 0
user_name = None
user_password = None
take_picture_event = threading.Event()
show_face_event = threading.Event()
stop_event = threading.Event()
face_det_event = threading.Event()

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure the upload folder exists

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

# @app.route('/')
# def index():
#     return '''
#         <h1>In progress</h1>
#         <p>Go to <a href="/upload">Upload</a> or <a href="/delete">Delete</a> or <a href="/recognition">Recognition</a></p>
#     '''


@app.route('/', methods=['GET'])
def recognition():
    flash('Opening Recognition Camera...')
    threading.Thread(target=FullFaceRecognitionBLE.main, args=(take_picture_event, show_face_event, face_det_event, stop_event)).start()
    return render_template('recognition.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global picture_index, take_picture_event, user_name, user_password
    if request.method == 'POST':
        user_name = request.form['user_name']
        user_password = request.form['password']

        if not user_name or not user_password:
            flash('User name and password are required.')
            return redirect(request.url)
        try:
            show_face_event.set()
            return redirect(url_for('camera'))
        except Exception as e:
            flash(f'Error: {e}')

        return redirect(url_for('upload'))

    return render_template('upload.html')

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    global picture_index, take_picture_event, user_name, user_password
    if request.method == 'POST':
        if face_det_event.is_set():
            picture_index += 1
            take_picture_event.set()
            if picture_index == 5:
                take_picture_event.clear()
                try:
                    time.sleep(.5)  # to makes sure all images are in /captures
                    show_face_event.clear()
                    ImagePreprocessing.main(user_name, user_password)
                    flash('Succesfully Processed and Uploaded User!')
                    picture_index = 0
                    return redirect(url_for('upload'))
                except Exception as ex:
                    flash(f'error: {ex}')
            else:
                flash(f'Photo taken {picture_index}')
        else:
            flash('No Face Detected In Frame, Please Try Again')
        return redirect(url_for('camera'))

    return render_template('camera.html')


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
    app.run(debug=False, port=1234)
