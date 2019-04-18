from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import json

import os
import urllib
import base64
from flask import Flask,request
from flask import Flask,render_template, redirect,url_for
import glob



app=Flask(__name__)

name_of_file="image"
ctr=0

path_to_known_faces='/home/sukrit/Desktop'


known_faces=[]
known_face_encodings=[]


#print "KnownFaces[]: ", known_faces

@app.route('/<bitmapString>')

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def detect_EAR():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
    args = vars(ap.parse_args())
 
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3


    COUNTER = 0
    TOTAL = 0


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
   # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    #print (lStart)	
    #print (lEnd)
    #print (rStart)	
    #print (rEnd)
    #print(face_utils.FACIAL_LANDMARKS_IDXS["mouth"])

    #print("[INFO] starting video stream thread...")
    vs = FileVideoStream(args["video"]).start()
    fileStream = True
    #vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    fileStream = False
    time.sleep(1.0)


    while True:
    	if fileStream and not vs.more():
		break

	frame = cv2.imread("a.PNG")
	#frame = vs.read()
	#print("Success till File Read!!!")
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
	rects = detector(gray, 0)
	detections = detector(gray, 0)
	for k,d in enumerate(detections): #For all detected face instances individually
	        shape = predictor(gray, d) #Draw Facial Landmarks with the predictor class
	        xlist = []
	        ylist = []
		landmarksX = []
		landmarksY = []
	        for i in range(0,68): #Store X and Y coordinates in two lists
	            xlist.append(float(shape.part(i).x))
	            ylist.append(float(shape.part(i).y))
	        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
	            landmarksX.append(x)
	            landmarksY.append(y)
#        if len(detections) > 0:
#		print(len(landmarksX))
#	        print(landmarksX)
#	        print(landmarksY)
	lMouth62_68 = dist.euclidean((landmarksX[61],landmarksY[61]),(landmarksX[67],landmarksY[67]) )
        lMouth64_66 = dist.euclidean((landmarksX[63],landmarksY[63]),(landmarksX[65],landmarksY[65]) )
        lMouth61_65 = dist.euclidean((landmarksX[60],landmarksY[60]),(landmarksX[64],landmarksY[64]) )

        mEar = (lMouth62_68 + lMouth64_66) / (2.0 * lMouth61_65) 

	print("MOR = ",mEar)

	NLR = dist.euclidean((landmarksX[27],landmarksY[27]),(landmarksX[30],landmarksY[30]) )

	print("NLR = ",NLR)

	for rect in rects:
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#lmouth = shape[51:59]
		#rmouth = shape[53:57]
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		#lmouthEar = eye_aspect_ratio(lmouth)
		#rmouthEar = eye_aspect_ratio(rmouth)
		#mouthEar = (lmouthEar + rmouthEar) / 2.0
		#print (mouthEar)
		
		ear = (leftEAR + rightEAR) / 2.0
		return_list=[]
		return_list.append(ear)
		return_list.append(mEar)
		return_list.append(NLR)
		data = {}
		data['ear'] = ear
		data['mor'] = mEar
		data['nlr'] = NLR
		json_data = json.dumps(data)
		return json_data
 
	if key == ord("q"):
		break
    cv2.destroyAllWindows()
    vs.stop()


def bitmapStr(bitmapString):
	bitmapString=bitmapString.replace(" ","")
	firstStr=bitmapString.replace('@','/')
	secondStr=firstStr.replace('&','+')
	missing_padding=len(bitmapString)%4

	file1=open(complete_name,"wb")
	file1.write(image_64_decode)
	file1.close()

	return 'Strnmn,m'

@app.route('/greet',methods=['GET','POST'])


def greet():
	a=request.form['greet']
	return a

@app.route('/',methods=['GET','POST'])

def index():
	imageFile = request.form["image"]
	counter = request.form["count"]
	personName = request.form["personName"]
	image_64_decode=base64.b64decode(imageFile)
	
	filepath = os.path.join(path_to_known_faces,"a"+".PNG")
	
	if not os.path.exists(path_to_known_faces):
        	os.makedirs(path_to_known_faces)   
	file1 = open(filepath, "wb")
	

	file1.write(image_64_decode)


	file1.close()
	
        # The Script for Ear Calculation Goes Here......
        #rvalue = str(detect_EAR())
	
       # print("EAR = ",rvalue)
	
	#return_list.append(rvalue)  #EAR appended
	
	ret = detect_EAR()
	print ret
        
#	return  '{} {} {}'.format(11,22,33)
        return ret
	

@app.route('/bacon',methods=['GET','POST'])
def bacon():
	if request.method=='POST':
		return 'Method used is POST'
	else:
		return 'Method used is GET'

if(__name__=="__main__"):
	app.run(debug=True,host=' 192.168.43.187',port=5002)

@app.route('/<user>')
def index(user=None):
	return render_template("user.html",user=user)

UPLOAD_FOLDER = '/home/sukrit/Desktop/FlaskBoston/UploadFiles'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER






