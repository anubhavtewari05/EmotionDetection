import os.path

from flask import Flask ,render_template,request,jsonify
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\Anubhav\PycharmProjects\Faceexp\haarcascade_frontal_face.xml')
model = load_model(r'C:\Users\Anubhav\PycharmProjects\Faceexp\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Sad','Neutral','Surprise']

picFolder = os.path.join('static')

app = Flask(__name__,template_folder='templates')

app.config['UPLOAD_FOLDER'] = picFolder


@app.route("/")
def index():
    pic1= os.path.join(app.config['UPLOAD_FOLDER'],'1.jpg')
    return render_template('home.html',image=pic1)

@app.route("/Aboutus")
def about():
    pic2 = os.path.join(app.config['UPLOAD_FOLDER'], '2.png')
    return render_template('Aboutus.html',image2= pic2)

@app.route("/OurProject")
def project():
    return render_template('OurProject.html')

@app.route("/capture")
def video_cap():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                preds = model.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

            else:
                cv2.putText(frame,'No Face Found',(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        cv2.imshow('Emotion Detector',frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    return project()

app.run(debug=True)