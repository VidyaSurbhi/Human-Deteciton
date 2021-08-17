import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import argparse 
import imutils
from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import time
from math import pow, sqrt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image
import tempfile






def mask_image():
    global RGB_img
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")

    # load the input image from disk and grab the image spatial
    # dimensions
    image = cv2.imread("./pics/withmaskimg.jpg")
    (h,w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
	
	 ##Compute the CNN out with the provided image
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def videopreds():
    global RGB_video
    video_file = st.file_uploader("Upload Video", type=['mp4']) 
    if video_file is None:
       st.write("Please Select File..!!!")
    elif video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())    # define a helper function to detected face and bounding box for each image 
    # in a live video frame
        def detect_and_predict_blood(frame, faceNet, maskNet):
            
            # grab the dimensions of the frame and then construct a blob
            # from it
            (h,w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()
            
            # initialize our list of faces, their corresponding locations,
            # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]
                
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > 0.5:
                    
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    
                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

                    # only make a predictions if at least one face was detected
            if len(faces) > 0:
                
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)
                
            # return a 2-tuple of the face locations and their corresponding
            # locations
            return (locs, preds)

        # load our serialized face detector model from disk
        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())
        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([args["face"],"deploy.prototxt"])
        weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        maskNet = load_model(args["model"])
        

        # initialize the video stream and allow the camera sensor to warm up
        # vs = VideoStream(src=0).start()
        
        # time.sleep(2.0)
        @st.cache(allow_output_mutation=True)
        def get_cap():
            return cv2.VideoCapture(tfile.name)

        cap = get_cap()
        
        # cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_movie = cv2.VideoWriter("./facevideoresult.avi", fourcc, 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        frameST = st.empty()
        #param=st.sidebar.slider('chose your value')

        # loop over the frames from the video stream
        while True:
            
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            # frame = vs.read()
            # frame = imutils.resize(frame, width=400)
            ret, frame = cap.read()
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_blood(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # determine the class label and color we'll use to draw
            # the bounding box and text
            
            
                
            

            # display the label and bounding box rectangle on the output
            # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF
            # RGB_video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # st.markdown('<h3 align="center">Video uploaded successfully!</h3>', unsafe_allow_html=True)
            # if st.button('Generate',key=8):
            #     st.video(RGB_video)
            
            frameST.image(frame, channels="BGR")
            
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break
                
        # do a bit of cleanup
        #cv2.destroyAllWindows()
        #vs.stop()

def videowebcam():
    # define a helper function to detected face and bounding box for each image 
    # in a live video frame
    def detect_and_predict(frame, faceNet, maskNet):
        
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        
        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                
                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

                # only make a predictions if at least one face was detected
        if len(faces) > 0:
            
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
            
        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # load our serialized face detector model from disk
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"],"deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])
    

    # initialize the video stream and allow the camera sensor to warm up
    # vs = VideoStream(src=0).start()
    
    # time.sleep(2.0)
    #@st.cache(allow_output_mutation=True)
    def get_cap():
        return cv2.VideoCapture(0)

    cap = get_cap()

    frameST = st.empty()
    #param=st.sidebar.slider('chose your value')

    # loop over the frames from the video stream
    while True:
        
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        # frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        ret, frame = cap.read()
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# determine the class label and color we'll use to draw
		# the bounding box and text
		
		
			
		

		# display the label and bounding box rectangle on the output
		# frame
                cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        frameST.image(frame, channels="BGR")
        
        # if the `q` key was pressed, break from the loop
        #if key == ord("q"):
        # break
            
    # do a bit of cleanup
    cv2.destroyAllWindows()
    

###################################################################################################################
# I used CLAHE preprocessing algorithm for detect humans better.
# HSV (Hue, Saturation, and Value channel). CLAHE uses value channel.
# Value channel refers to the lightness or darkness of a colour. An image without hue or saturation is a grayscale image.
 #Constant Values
 #Constant Values
preprocessing = False
calculateConstant_x = 300
calculateConstant_y = 615
personLabelID = 15.00
debug = True
accuracyThreshold = 0.4
RED = (0,0,255)
YELLOW = (0,255,255)
GREEN = (0,255,0)
write_video = False
def CLAHE(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def centroid(startX,endX,startY,endY):
    centroid_x = round((startX+endX)/2,4)
    centroid_y = round((startY+endY)/2,4)
    bboxHeight = round(endY-startY,4)
    return centroid_x,centroid_y,bboxHeight

def calcDistance(bboxHeight):
    distance = (calculateConstant_x * calculateConstant_y) / bboxHeight
    return distance

def drawResult(frame,position,highRisk,mediumRisk,detectionCoordinates):
    for i in position.keys():
        if i in highRisk:
            rectangleColor = RED
        elif i in mediumRisk:
            rectangleColor = YELLOW
        else:
            rectangleColor = GREEN
        (startX, startY, endX, endY) = detectionCoordinates[i]

        cv2.rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)


def social_video():
    frameST = st.empty()
    social_file = st.file_uploader("Upload Video", type=['mp4']) 
    if social_file is None:
       st.write("Please Select File..!!!")
    elif social_file is not None:
        sfile = tempfile.NamedTemporaryFile(delete=False)
        sfile.write(social_file.read())  
        caffeNetwork = cv2.dnn.readNetFromCaffe("./social_dist/SSD_MobileNet_prototxt.txt", "./social_dist/SSD_MobileNet.caffemodel")
        cap = cv2.VideoCapture(sfile.name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_movie = cv2.VideoWriter("./socialresult.avi", fourcc, 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


        while cap.isOpened():

            debug_frame, frame = cap.read()
            highRisk = set()
            mediumRisk = set()
            position = dict()
            detectionCoordinates = dict()

            if not debug_frame:
                print("Video cannot opened or finished!")
                break

            if preprocessing:
                frame = CLAHE(frame)

            (imageHeight, imageWidth) = frame.shape[:2]
            pDetection = cv2.dnn.blobFromImage(cv2.resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)

            caffeNetwork.setInput(pDetection)
            detections = caffeNetwork.forward()

            for i in range(detections.shape[2]):

                accuracy = detections[0, 0, i, 2]
                if accuracy > accuracyThreshold:
                    # Detection class and detection box coordinates.
                    idOfClasses = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                    (startX, startY, endX, endY) = box.astype('int')

                    if idOfClasses == personLabelID:
                        # Default drawing bounding box.
                        bboxDefaultColor = (255,255,255)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), bboxDefaultColor, 2)
                        detectionCoordinates[i] = (startX, startY, endX, endY)

                        # Centroid of bounding boxes
                        centroid_x, centroid_y, bboxHeight = centroid(startX,endX,startY,endY)                    
                        distance = calcDistance(bboxHeight)
                        # Centroid in centimeter distance
                        centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
                        centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y
                        position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)

            #Risk Counter Using Distance of Positions
            for i in position.keys():
                for j in position.keys():
                    if i < j:
                        distanceOfBboxes = sqrt(pow(position[i][0]-position[j][0],2) 
                                            + pow(position[i][1]-position[j][1],2) 
                                            + pow(position[i][2]-position[j][2],2)
                                            )
                        if distanceOfBboxes < 150: # 150cm or lower
                            highRisk.add(i),highRisk.add(j)
                        elif distanceOfBboxes < 200 > 150: # between 150 and 200
                            mediumRisk.add(i),mediumRisk.add(j) 
        

            cv2.putText(frame, "Person in High Risk : " + str(len(highRisk)) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Person in Medium Risk : " + str(len(mediumRisk)) , (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "Detected Person : " + str(len(detectionCoordinates)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            drawResult(frame, position,highRisk,mediumRisk,detectionCoordinates)
            # if write_video:            
            #     output_movie.write(frame)
            #cv2.imshow('Result', frame)
            frameST.image(frame, channels="BGR")
            #waitkey = cv2.waitKey(1)
            #if waitkey == ord("q"):
            # break

        #cap.release()
            cv2.destroyAllWindows()

##############################################################################
def main():
    st.title("Face Mask Detector & Social Distancing")
    st.text("Detecting Using OpenCV , TensorFlow & Keras")
    choice_detect = st.sidebar.selectbox("Select Detect Module",("FaceMask","SocialDistance"),key="1")

    if choice_detect =='FaceMask':
        choice = st.sidebar.selectbox("Detection Mode",("Image","Video","Webcam"),key="2")
        if choice == 'Image':
            st.subheader("Detect on Image")
            mask_image()  
            image_file = st.file_uploader("", type=['jpg'])  # upload image
            if image_file is not None:
                our_image = Image.open(image_file)  # making compatible to PIL
                im = our_image.save('./pics/withmaskimg.jpg')
                saved_image = st.image(image_file, caption='', use_column_width=True)
                st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
                if st.button('Process',key="11"):
                    st.image(RGB_img, use_column_width=True)
            
            enhance_type = st.sidebar.radio("Enchance type", ("Original","Gray-scale","Contrast","Brightness","Blurring"),key="5")
            if enhance_type == 'Gray-scale':
                # new_img = np.array(RGB_img.convert('RGB'))
                img = cv2.cvtColor(RGB_img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray)

        elif choice == 'Video':
            st.subheader("Detect in Video")
            videopreds()
            # if video_file is not None:
            #     videopreds()
                #video_byte = open(video_file, 'rb').read()
                #saved_video = st.video(video_byte)
                

        elif choice == 'Webcam':
            st.subheader("Detect in Webcam")
            videowebcam()

    elif choice_detect == 'SocialDistance':
        choice = st.sidebar.selectbox("Detection Mode",("Image","Video"),key="3")
        if choice == 'Video':
            social_video()


if __name__ == '__main__':
    main()



        
       
