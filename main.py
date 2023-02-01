from asyncio.windows_events import NULL
import os
import string
from tokenize import String
from flask import Flask, render_template, request, redirect, Response, url_for, make_response

import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import cv2
from tensorflow.keras.utils import img_to_array
import os
import re
import torch.cuda
from datetime import datetime
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load the leaf detection model from disk
model = load_model("./predictionModel.model")
directory_colourIMG = r'./static/coloredImages'
project_directory = os.getcwd()
SIZE = 160


app = Flask(__name__)



def coloringImage(path):
    try:
        img = cv2.imread(path,1)
        
            #Resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        grey = img.astype('float32') / 255.0

        predicted = np.clip(model.predict(grey.reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)

    
        # os.chdir(directory_colourIMG)
        datetimeconersion = str(datetime.today().timestamp())
        
        filename = 'savedImage'+ datetimeconersion +'.png'
        plt.imshow(predicted)
        plt.axis('off')
        plt.savefig(directory_colourIMG+'/'+filename, bbox_inches='tight', pad_inches=0,transparent=True)

        
        print(directory_colourIMG+'/'+filename)
    except Exception as e:
        print(str(e))
    # os.chdir(project_directory)
    return directory_colourIMG+'/'+filename


    
@app.route('/')
@app.route('/home')
def home():
    image = NULL
    message = ''
    image =  request.args.get('image')
    message = request.args.get('message')
    return render_template('colouringMain.html', image = image, message=message)

UPLOAD_FOLDER = './static/uploadedImages'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif','jfif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploadImage',  methods=['POST'])
def uploadImage():
    file = ''
    filepath=''
    image= NULL
    msg=''
    file = request.files['formFile']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER']+'/'+ filename
        print(filepath)
        file.save(filepath)
        
        image = coloringImage(filepath)
    else:
        msg = 'file type not supported'
    
    return redirect(url_for('home', image = image, message = msg))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
