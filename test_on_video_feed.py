import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
 
CATEGORIES =[ 'paper','rock', 'scissor']

# Loading the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1)
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    IMG_SIZE = 150
    img_array = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY) 
    new_array = cv2.resize(blackAndWhiteImage, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    
    cv2.imshow("roi", roi)
    
    prediction = loaded_model.predict(new_array)
    print(CATEGORIES[np.argmax(prediction[0])])
    
    cv2.imshow('gray', frame)
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



