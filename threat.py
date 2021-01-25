
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import imutils 
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
model = keras.models.load_model("model_35_91_61.h5")
face_cas = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
gun_cascade = cv.CascadeClassifier('cascade.xml')
font = cv.FONT_HERSHEY_SIMPLEX


emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


inWidth = args.width
inHeight = args.height


cap = cv.VideoCapture(0)


firstFrame = None
gun_exist = False


while True:
	# Processing frame here
  hasFrame, frame = cap.read()
  if not hasFrame:
  	print("Frame not found. Exiting")
  	break
  

  frameWidth = frame.shape[1]
  frameHeight = frame.shape[0]

  # PRocessing for pose from here
  net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
  out = net.forward()
  out = out[:, :19, :, :]

  assert(len(BODY_PARTS) == out.shape[1])

  points = []
  for i in range(len(BODY_PARTS)):
      # Slice heatmap of corresponging body's part.
      heatMap = out[0, i, :, :]

      
      _, conf, _, point = cv.minMaxLoc(heatMap)
      x = (frameWidth * point[0]) / out.shape[3]
      y = (frameHeight * point[1]) / out.shape[2]
      # Add a point if it's confidence is higher than threshold.
      points.append((int(x), int(y)) if conf > args.thr else None)

  for pair in POSE_PAIRS:
      partFrom = pair[0]
      partTo = pair[1]
      assert(partFrom in BODY_PARTS)
      assert(partTo in BODY_PARTS)

      idFrom = BODY_PARTS[partFrom]
      idTo = BODY_PARTS[partTo]

      if points[idFrom] and points[idTo]:
          cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
          cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
          cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

  t, _ = net.getPerfProfile()
  freq = cv.getTickFrequency() / 1000
  cv.putText(frame, '%.2fms' % (t / freq), (10, 20), font, 0.5, (0, 0, 0))
  	

  # # PRocessing for expression from here
  
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  faces = face_cas.detectMultiScale(gray, 1.3,5)
  for (x, y, w, h) in faces:
  	face_component = gray[y:y+h, x:x+w]
  	fc = cv.resize(face_component, (48, 48))
  	inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
  	inp = inp/255.
  	prediction = model.predict(inp)
  	em = emotion[np.argmax(prediction)]
  	score = np.max(prediction)
  	cv.putText(frame, em, (x, y), font, 0.5, (0, 255, 0), 2)
  	cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


  # PRocessing for gun from here
   
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
  
  gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100)) 
  
  if len(gun) > 0: 
    gun_exist = True
    
  for (x, y, w, h) in gun: 
    
    frame = cv.rectangle(frame, 
              (x, y), 
              (x + w, y + h), 
              (255, 0, 0), 2)
    cv.putText(frame, 'gun', (x, y-10), font, 0.5, (36,255,12), 2) 
    roi_gray = gray[y:y + h, x:x + w] 
    roi_color = frame[y:y + h, x:x + w]  

  if firstFrame is None: 
    firstFrame = gray 
    continue

  
  cv.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"), 
        (10, frame.shape[0] - 10), 
        font, 
        0.35, (0, 0, 255), 1) 

  cv.imshow("Result", frame) 
  key = cv.waitKey(1) & 0xFF
  
  if key == ord('q'):
  	break