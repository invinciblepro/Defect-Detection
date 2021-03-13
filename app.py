import io
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow.keras
from PIL import Image, ImageOps

UPLOAD_FOLDER = 'user_input'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    im=request.files['test']
    im.save(os.path.join(app.config['UPLOAD_FOLDER'], "test.jpg"))
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('user_input/test.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    text=""
    if(prediction[0][0]>prediction[0][1]):
      text="Defective piece"
    else:
      text="Ok piece"
    return render_template('index.html', prediction_text='Result obtained : {}'.format(text), url="user_input/test.jpg")

if __name__ == "__main__":
    app.run(debug=True)