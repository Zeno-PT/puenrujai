import cv2
import numpy as np
import json
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response, request, redirect, url_for
from PIL import Image

# Initialize the Flask app
app = Flask('puenrujai-v1')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']
# camera = cv2.VideoCapture(0)
labels = []
count = 0


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def gen_img(cam):
    frame = load_image_into_numpy_array(cam)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48),
                              interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            #label = 'Sad'

            print(label)
            labels.append(label)
            label_position = (x, y)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    frame = frame.decode('ISO-8859-1')
    frame = json.dumps(frame)
    return frame


def show_labels(la):
    if len(la) > 4:
        b = la[-4:]
    else:
        b = la
    ans = ""
    for i in b:
        ans = ans + i + " "
    labels.clear()
    return ans


@app.route('/', methods=['GET'])
def hello():
    return render_template('hello.html')


@app.route('/hello2', methods=['GET'])
def hello2():
    # labels.clear()
    return render_template('hello2.html')


@app.route('/name', methods=['GET', 'POST'])
def name():
    if request.method == 'GET':
        return render_template('name.html')
    else:
        nm = request.form['name']
        return render_template('name2.html', value=nm)


@app.route('/name2', methods=['GET'])
def name2():
    return render_template('name2.html')


@app.route('/main', methods=['GET'])
def main():
    return render_template('main.html')


@app.route('/main2', methods=['GET'])
def main2():
    return render_template('main2.html')


@app.route('/main3', methods=['GET'])
def main3():
    return render_template('main3.html')


@app.route('/main4', methods=['GET'])
def main4():
    return render_template('main4.html')


@app.route('/main5', methods=['GET'])
def main5():
    return render_template('main5.html')


@app.route('/main6', methods=['GET'])
def main6():
    return render_template('main6.html')


@app.route('/local', methods=['GET'])
def local():
    if request.method == 'GET':
        ans = show_labels(labels)
        return render_template('local.html', value=ans)
    # else:
        # ans = show_labels(labels)
        # ans2 = most_frequent(labels)
        # return render_template('local.html', value=ans)


@app.route('/demo', methods=['GET'])
def demo():
    if request.method == 'GET':
        ans = show_labels(labels)
        return render_template('demo.html', value=ans)
    # else:
        # ans = show_labels(labels)
        # ans2 = most_frequent(labels)
        # return render_template('demo.html', value=ans)


@app.route('/image', methods=['POST'])
def image():
    image_file = request.files['image']  # get the image
    image_object = Image.open(image_file)
    objects = gen_img(image_object)
    return objects


@app.route('/menu', methods=['GET'])
def menu():
    return render_template('menu.html')


@app.route('/community', methods=['GET'])
def community():
    return render_template('community.html')


@app.route('/pet', methods=['GET'])
def pet():
    return render_template('pet.html')


@app.route('/firework1', methods=['GET'])
def firework1():
    return render_template('firework1.html')


@app.route('/firework2', methods=['GET'])
def firework2():
    return render_template('firework2.html')


@app.route('/firework3', methods=['GET'])
def firework3():
    return render_template('firework3.html')


@app.route('/record', methods=['GET'])
def record():
    return render_template('record.html')


'''
@app.route('/video-result', methods=['GET'])
def video_feed():
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera-cat', methods=['POST','GET'])
def getCameraCat():
    if request.method=='GET':
        return render_template('camera-cat.html')
    else:
#        detectedImg = request.form.get('videoElement')
        while True:
            return render_template('result-cat.html',value=show_labels(),ex=url_for("video_feed"))
'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
