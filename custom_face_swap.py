import cv2
import numpy as np
import dlib

# Initialize detector and predictor for faces in images
frontal_face_detector = dlib.get_frontal_face_detector()
# Using a pretrained model a facial landmarks
frontal_face_predictor = dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")


# Read src and dest images and covert to grayscale
src_img = cv2.imread('images/ayobami.jpg')
src_img_ = src_img
src_imgGrey = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)


dest_img = cv2.imread('images/ayodeji.jpg')
dest_img_ = dest_img
dest_imgGrey = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)


# Create zeros array canvas for src and dest images
src_imgCanvas = np.zeros_like(src_imgGrey)

height, width, channels = dest_img.shape
dest_imgCanvas = np.zeros((height, width, channels), np.uint8)

# Find the faces in src_img
#Returns a numpy array of
src_faces = frontal_face_detector(src_imgGrey)

# Loop through all faces found in src_img
for src_face in src_faces:
    #Predictor takes human face as input and returns the list of facial landmarks
    src_face_landmarks = frontal_face_predictor(src_imgGrey, src_face)
    src_face_landmark_points = []
    
    # Loop through all the 68 landmark points and add them to the tuple
    for landmark_no in range(0, 68):
        x_point = src_face_landmarks.part(landmark_no).x
        y_point = src_face_landmarks.part(landmark_no).y
        src_face_landmark_points.append((x_point, y_point))
    print(src_face_landmark_points)
    
    
    
