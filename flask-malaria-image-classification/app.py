import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow import keras
import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
model_dict = {'Native CNN'   :   'static\MLModule\cnnModel.h5',
              'Transfer Learning'     :   'static\MLModule\XceptionModel.h5',}
              
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('/select.html', )

@app.route('/predict', methods=['POST'])
def predict():
    if request.form['action'] == 'predict':
        print("---------------------PREDICT--------------------")
        chosen_model = request.form['select_model']

        if chosen_model in model_dict:
            model = keras.models.load_model(model_dict[chosen_model]) 
        else:
            model = keras.models.load_model(model_dict[0])
        file = request.files["file"]
        file.save(os.path.join('static', 'temp.jpg'))
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
        img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        start = time.time()
        pred = model.predict(img)[0]
        label = np.where(pred > 0.5, 1,0).astype(int)
#        acc = 100*tf.reduce_max(pred)
#        print('----------PRED------------',pred)
#        print('-------------ACC------------- : {}% '.format(100-acc))
        runtimes = round(time.time()-start,4)
        respon_model = [round(elem * 100, 2) for elem in pred]
        print(chosen_model)
        return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg', label)

    elif request.form['action'] == 'compare':
        print("---------------------COMPARE--------------------")
        model_1 = keras.models.load_model(model_dict['Native CNN'])
        model_2 = keras.models.load_model(model_dict['Transfer Learning'])
        file = request.files["file"]
        file.save(os.path.join('static', 'temp.jpg'))
        model_s = [model_1, model_2]
        label_s = []
        runtimes_s = []
        respon_model_s = []
#       LOOP for
        for x in range(2):
            img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
            img = np.expand_dims(cv2.resize(img, model_s[x].layers[0].input_shape[0][1:3] if not model_s[x].layers[0].input_shape[1:3] else model_s[x].layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
            start = time.process_time()
            pred = model_s[x].predict(img)[0]
            end = time.process_time()
            label_s.append(np.where(pred > 0.5, 1,0).astype(int))
            runtimes_s.append((end-start))
            respon_model_s.append([round(elem * 100, 2) for elem in pred])

        return compare_result(list(model_dict)[0], list(model_dict)[1], label_s[0], label_s[1], runtimes_s[0], runtimes_s[1], respon_model_s[0], respon_model_s[1], 'temp.jpg')


def predict_result(model, run_time, probs, img, label):
    class_list = {0 : 'Parasitized', 1 : 'Uninfected'}
    labels = class_list.get(label[0])
    if label == 0:
        probs = 100 - probs[0]
    elif label == 1:
        probs = probs[0]
    return render_template('/result_select.html', labels=labels, 
                            probs=probs, model=model,
                            run_time=run_time, img=img)

def compare_result(model1, model2, label1, label2 , runtime1, runtime2, prob1, prob2, img):
    class_list = {0 : 'Parasitized', 1 : 'Uninfected'}
    labels_1 = class_list.get(label1[0])
    labels_2 = class_list.get(label2[0])
    if label1 == 0:
        probs_1 = 100 - prob1[0]
    elif label1 == 1:
        probs_1 = prob1[0]
    if label2 == 0:
        probs_2 = 100 - prob2[0]
    elif label1 == 1:
        probs_2 = prob2[0]
    return render_template('/result_compare.html', label1=labels_1, label2=labels_2, prob1=probs_1,
                            prob2=probs_2, model1=model1, model2=model2, runtime1=runtime1, 
                            runtime2=runtime2, img=img)   




if __name__ == "__main__": 
        app.run(debug=True, host='0.0.0.0', port=2000)
