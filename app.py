# from fileinput import filename
from flask import Flask, render_template, request, flash, request, redirect, url_for, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import urllib.request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = r"C:\Users\Akshay Deshmukh\Desktop\Data Science\Deep Learning\Deep_Learning_A_Z\Image Classification Deployment\uploads"


# APP_ROOT = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = '/uploads/'
# UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/..')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


cnn = load_model(r'C:/Users/Akshay Deshmukh/Desktop/Data Science/Deep Learning/Deep_Learning_A_Z/Image Classification Deployment/env/cnn.h5')

cnn.make_predict_function()

def pred(img_path):
	test_image = image.load_img(img_path, target_size = (64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = cnn.predict(test_image)
	if result[0][0] == 1:
		prediction="Dog"
	else:
		prediction="Cat"
	return prediction



@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('index.html')

# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
#             return redirect(url_for('download_file', name=filename))
#     return render_template('index.html')




@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['file']

		img_path = "C:/Users/Akshay Deshmukh/Desktop/Data Science/Deep Learning/Deep_Learning_A_Z/Supervised Deep Learning/Convolutional Neural Networks/dataset/single_prediction/" + img.filename	
		img.save(img_path)

		# img_path = app.config['UPLOAD_FOLDER'] + secure_filename(img.filename)
		print(img_path)

		p = pred(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)