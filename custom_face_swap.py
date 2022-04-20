import numpy as np
import cv2
import face_recognition
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor('dataset/shape_predictor_68_face_landmarks.dat')

# Read the source and dest image
src_img = cv2.imread('images/ayobami.JPG')
src_img_cpy = src_img.copy()
src_img_grey = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

dest_img = cv2.imread('images/ayodeji.jpg')
dest_img_cpy = dest_img.copy()
dest_img_grey = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)

# Black Image canvas for src-img_grey
src_img_canvas = np.zeros_like(src_img_grey)

# Black image_camvas for dest_img
h, w, nC = dest_img.shape
dest_img_canvas = np.zeros((h, w, nC), np.uint8)

#Find the faces in src_img
src_faces = frontal_face_detector(src_img_grey)
for src_face in src_faces:
        src_face_landmarks = frontal_face_predictor(src_img_grey, src_face)
        src_face_landmark_pts = []

        # Extract the landmark points and plot them on the image
        for i in range(0, 68):
                x_point = src_face_landmarks.part(i).x
                y_point = src_face_landmarks.part(i).y
                src_face_landmark_pts.append((x_point, y_point))
                #cv2.circle(src_img, (x_point, y_point), 3, (255, 0, 0), -1)
                # cv2.imshow('img', src_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        # FIND THE CONVEXHULL OF THE IMAGE
        #Convert the landmark points to numpy array
        src_face_landmark_pts_arr = np.array(src_face_landmark_pts, np.int32)
        src_face_convexhull = cv2.convexHull(src_face_landmark_pts_arr)
        # Display convex hull
        cv2.polylines(src_img, [src_face_convexhull], True, (255, 0, 0), 3)
        #draw the filled polygon along the hull over the zero array canvas
        cv2.fillConvexPoly(src_img_canvas, src_face_convexhull, 255)
        #Place over the src_img canvas
        src_face_image = cv2.bitwise_and(src_img, src_img, mask=src_img_canvas)

cv2.imshow('img', src_face_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(src_img_canvas.shape)





































# import cv2
# import numpy as np
# import dlib

# # Initialize detector and predictor for faces in images
# frontal_face_detector = dlib.get_frontal_face_detector() #detect front any looking face
# # Using a pretrained model a facial landmarks
# frontal_face_predictor = dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")


# # Read src and dest images and covert to grayscale
# src_img = cv2.imread('images/ayobami.jpg')
# src_img_ = src_img
# src_imgGrey = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)


# dest_img = cv2.imread('images/ayodeji.jpg')
# dest_img_ = dest_img
# dest_imgGrey = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)


# # Create zeros array canvas for src and dest images
# src_imgCanvas = np.zeros_like(src_imgGrey)

# height, width, channels = dest_img.shape
# dest_imgCanvas = np.zeros((height, width, channels), np.uint8)

# # Find the faces in src_img
# #Returns a numpy array of
# src_faces = frontal_face_detector(src_imgGrey)

# # Loop through all faces found in src_img
# for src_face in src_faces:
#     #Predictor takes human face as input and returns the list of facial landmarks
#     src_face_landmarks = frontal_face_predictor(src_imgGrey, src_face)
#     src_face_landmark_points = []
    
#     # Loop through all the 68 landmark points and add them to the tuple
#     for landmark_no in range(0, 68):
#         x_point = src_face_landmarks.part(landmark_no).x
#         y_point = src_face_landmarks.part(landmark_no).y
#         src_face_landmark_points.append((x_point, y_point))
#     print(src_face_landmark_points)
    
    
    
