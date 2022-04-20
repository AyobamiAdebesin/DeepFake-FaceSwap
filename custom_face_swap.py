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
        #cv2.polylines(src_img, [src_face_convexhull], True, (255, 0, 0), 3)
        #draw the filled polygon along the hull over the zero array canvas
        cv2.fillConvexPoly(src_img_canvas, src_face_convexhull, 255)       
        #Place over the src_img canvas
        src_face_image = cv2.bitwise_and(src_img, src_img, mask=src_img_canvas)

# Find the Delaunay Triangulation Indices of src img
###################################################
        #Draw an approximate rectangle around the image
        bounding_rect = cv2.boundingRect(src_face_convexhull)

        # Create a Delaunay subdivision
        sub = cv2.Subdiv2D(bounding_rect)
        sub.insert(src_face_landmark_pts)
        triangle_vect = sub.getTriangleList()
        triangle_arr = np.array(triangle_vect, dtype=np.int32)
        print(triangle_arr)

        for tri in triangle_arr:
               index_point1 = (tri[0], tri[1])
               index_point2 = (tri[2], tri[3])
               index_point3 = (tri[4], tri[5])

               line_color = (255, 255, 255)
               cv2.line(src_img, index_point1, index_point2, line_color, 1)
               cv2.line(src_img, index_point2, index_point3, line_color, 1)
               cv2.line(src_img, index_point3, index_point1 , line_color, 1)
cv2.imshow('img', src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output/ayo.jpg', src_img)

















# cv2.imshow('img', src_face_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(src_img_canvas.shape)