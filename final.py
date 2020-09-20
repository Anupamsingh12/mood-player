import cv2
import numpy as np
from keras.models import load_model

from pygame import mixer

def start():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    video_capture = cv2.VideoCapture(0)      #for cam of laptop.....
    model = load_model('keras_model/model_5-49-0.62.hdf5')

    target = ['angry','disgust','fear','happy','sad','surprise','neutral']
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame=cv2.flip(frame,1,0)  #flip to act as mirror

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1)

    # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2,5)
            face_crop = frame[y:y+h,x:x+w]
            face_crop = cv2.resize(face_crop,(48,48))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = face_crop.astype('float32')/255
            face_crop = np.asarray(face_crop)
            face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1])
            result = target[np.argmax(model.predict(face_crop))]
            cv2.putText(frame,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)

    # Display the resulting frame
    video_capture.release()
    print('playing song for')
    print(result)
    if(result=="happy"):
        play_()
    #displays the final expression u had bwfore exiting....
    cv2.destroyAllWindows()

    
    cv2.imshow('Video', frame)
    

    if cv2.waitKey(10) ==27:
        break
inp=input("enter any key to start")
if True:
    start()
