import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

########################################## read and display image
#img = cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg')

#cv.imshow('img', img)

#cv.waitKey(0)


######################################## reading videos
#capture = cv.VideoCapture(0) # for web cam
#capture = cv.VideoCapture('C:/Users/bcfri/OneDrive/Videos/dog.mp4') # for file video

#while True:
#        isTrue, frame = capture.read()
#        cv.imshow('Video',frame)

#        if cv.waitKey(20) & 0xFF==ord('d'):
#            break

#capture.release()
#cv.destroyAllWindows()

#cv.waitKey(0)


############################################# resizing and rescaling

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame,dimensions, interpolation = cv.INTER_AREA)

#img = cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg')
#img_resized = rescaleFrame(img,1.25)
#cv.imshow('img', img_resized)
#cv.waitKey(0)

### for videos only
def changeRes(width,height):
    #only for live cam footage, not stored videos
    cv.capture.set(3,width)
    cv.capture.set(4,height)

############################################# Drawing Shapes & Placing Text

#blank = np.zeros((500,500,3), dtype='uint8') # create a blank image of 500 by 500 pixles
#cv.imshow('blank',blank)

##change the color
#blank[:] = 0,255,0
#cv.imshow('Green', blank)

#blank[200:300, 300:400] = 0,0,255
#cv.imshow('partial red', blank)

##draw a rectangle
#cv.rectangle(blank, (20,20), (100,100), (255,0,0), thickness=2) #thickness = cv.FILLED to make the rectangle a solid color
#cv.imshow('rectangle', blank)

##draw a circle
#cv.circle(blank, (blank.shape[1]//2,blank.shape[0]//2), 40, (255,0,0), thickness = 3)
#cv.imshow('circle', blank)

##draw a line
#cv.line(blank, (200,120), ( 20, 300), (255,255,255), thickness = 3)
#cv.imshow('line', blank)

##draw text on image
#cv.putText(blank, 'Hello World!', (100,400), cv.FONT_HERSHEY_TRIPLEX, scale = 2.0, color = (0,0,0), thickness = 1)
#cv.imshow('text', blank)

#cv.waitKey(0)

############################################# Basic Functions


# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# #convert image to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# #Blur
# blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT) ## file, kernal(higher is more blur, must be odd), sigmax
# cv.imshow('Blur', blur)

# #edge cascade
# canny = cv.Canny(blur, 125, 175)
# cv.imshow('canny',canny)

# #Dilating the image
# dilated = cv.dilate(canny, (3,3), iterations = 2)
# cv.imshow('dilated',dilated)

# #eroding
# eroded = cv.erode(dilated, (3,3), iterations = 2)
# cv.imshow('eroded', eroded)

# #resize
# resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA) #INTER_LINAR/INTER_CUBIC for upsizing (cubic is better but slower)
# cv.imshow('resized', resized)

# #cropping
# cropped = img[50:200, 200:400]
# cv.imshow('cropped', cropped)

# cv.waitKey(0) 

############################################# transformations

# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# #translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# # -x --> left
# # -y --> up
# # +x --> right
# # +y --> down

# translated = translate(img, 100, 100)
# cv.imshow('translated', translated)

# #Rotation
def rotate(img, angle, rotPoint = None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions)

# rotated = rotate(img,-45)
# cv.imshow('rotated',rotated)

# #Flipping
# flip = cv.flip(img, -1)
# cv.imshow('flipped',flip)

# cv.waitKey(0)

# ############################################# Contour Detection

# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# blank = np.zeros(img.shape, dtype='uint8')
# cv.imshow('blank', blank)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

# blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
# cv.imshow('blur', blur)

# canny = cv.Canny(blur,125,175)
# cv.imshow('canny', canny)

# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(f'{len(contours)} contour(s) found')

# cv.drawContours(blank, contours, -1, (0,0,255), 1)
# cv.imshow('canny contours drawn', blank)

# #another way to find contours
# ret,thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('thresh', thresh)

# contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(f'{len(contours)} contour(s) found')

# cv.drawContours(blank, contours, -1, (0,0,255), 1)
# cv.imshow('contours drawn', blank)


# cv.waitKey(0) 


############################################## COLOR SPACES

# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# #BGR to Grayscale]
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)

# #BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('hsv', hsv)


# #BGR to LAB
# lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('lab', lab)

# #BGR to RGB
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB', rgb)

# # plt.imshow(rgb)
# # plt.show()

# #hsv to BGR
# hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# cv.imshow('hsv -> bgr', hsv_bgr)

# #lab to BGR
# lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
# cv.imshow('lab -> bgr', lab_bgr)

# #gray to BGR
# gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) #still looks gray
# cv.imshow('gray -> bgr', gray_bgr)

# cv.waitKey(0)

############################################## COLOR CHANNELS

# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# blank = np.zeros(img.shape[:2], dtype='uint8')

# b,g,r = cv.split(img)

# blue = cv.merge([b,blank,blank])
# green = cv.merge([blank,g,blank])
# red = cv.merge([blank,blank,r])

# cv.imshow('blue', blue)
# cv.imshow('green', green)
# cv.imshow('red', red)

# print(img.shape)
# print(b.shape)
# print(g.shape)
# print(r.shape)

# merged = cv.merge([b,g,r])
# cv.imshow('merged image', merged)

# cv.waitKey(0)

############################################## BLURRING TECHNIQUES

# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# #averaging
# average = cv.blur(img, (3,3))
# cv.imshow('average blur', average)

# #gaussian blur
# gaussian = cv.GaussianBlur(img,(3,3),0)
# cv.imshow('gaussian blur', gaussian)

# #median blur
# median = cv.medianBlur(img, 3)
# cv.imshow('median blur', median)

# #bilateral blur
# bilateral = cv.bilateralFilter(img, 5, 35, 25)
# cv.imshow('bilateral blur',bilateral)

# cv.waitKey(0) 

############################################## BITWISE OPERATORS

# blank = np.zeros((400,400), dtype='uint8')

# rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
# circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

# cv.imshow('Rectangle', rectangle)
# cv.imshow('circle', circle)

# #bitwise AND
# bitwise_and = cv.bitwise_and(rectangle,circle)
# cv.imshow('bitwise and', bitwise_and)

# #bitwise or
# bitwise_or = cv.bitwise_or(rectangle, circle)
# cv.imshow('bitwise or', bitwise_or)

# #bitwise xor
# bitwise_xor = cv.bitwise_xor(rectangle, circle)
# cv.imshow('bitwise xor', bitwise_xor)


# #bitwise not
# bitwise_not = cv.bitwise_not( bitwise_xor )
# cv.imshow('bitwise xor not', bitwise_not)

# cv.waitKey(0) 

############################################## MASKING
# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow('Blank Image', blank)

# mask = cv.circle(blank, (img.shape[1]//2 -80, img.shape[0]//2 +45), 100, 255, -1)
# cv.imshow('mask', mask)

# masked = cv.bitwise_and(img, img, mask = mask)
# cv.imshow('masked', masked)

# cv.waitKey(0) 

############################################## COMPUTING HISTOGRAMS

# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# #for grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])
# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

# blank = np.zeros(img.shape[:2], dtype='uint8')
# mask = cv.circle(blank, (img.shape[1]//2 , img.shape[0]//2), 100, 255, -1)
# cv.imshow('mask', mask)
# masked = cv.bitwise_and(gray, gray, mask = mask)


# gray_hist = cv.calcHist([gray], [0], masked, [256], [0,256])
# plt.figure()
# plt.title('Grayscale Masked Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

# plt.figure()
# plt.title('color Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# #color histogram
# colors=('b','g','r')
# for i,col in enumerate(colors):
#     color_hist = cv.calcHist([img], [i], masked, [256], [0,256])
#     plt.plot(color_hist, color = col)
#     plt.xlim([0,256])

# plt.show()

# cv.waitKey(0) 

############################################## THRESHOLDING
# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# #simple thresholding
# threshold, thresh = cv.threshold(gray,100, 255, cv.THRESH_BINARY)
# cv.imshow('Simpl Thresholded', thresh)

# threshold, thresh_inverse = cv.threshold(gray,100, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Simpl Thresholded Inverse', thresh_inverse)

# #adaptive thresholding
# adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)
# cv.imshow('Adaptive Thresholding', adaptive_thresh)

# cv.waitKey(0) 

############################################## Edge Detection
# img = rescaleFrame(cv.imread('C:/Users/bcfri/OneDrive/Pictures/mar-bustos-HsEz1XZ1TO8-unsplash.jpg'), .15)
# cv.imshow('img', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# # Laplacian
# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

# # Sobel
# sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
# sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
# combined_sobel = cv.bitwise_or(sobelx, sobely)

# cv.imshow('Sobel x', sobelx)
# cv.imshow('Sobel y', sobely)
# cv.imshow('Sobel combined', combined_sobel)

# canny = cv.Canny(gray, 150, 175)
# cv.imshow('Canny', canny)

# cv.waitKey(0) 
############################################## Face Detection with haar cascade

# img = cv.imread('ultimate_group.jpg')
# cv.imshow('img', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray) 

# haar_cascade = cv.CascadeClassifier('haar_face.xml')

# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=6)

# print(f'Number of faces found = {len(faces_rect)}')

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness = 2)

# cv.imshow('detected faces', img)

# cv.waitKey(0) 

############################################## Face Recognition
################# train model
# people = ['Name1', 'Name2', 'etc']

# ####one way
# p = []
# for i in os.listdir(r'C:/Users/bcfri/OneDrive/Desktop/practice code/opencv_tutorial/Faces/train'):
#     p.append(i)


# #### another way
# DIR = r'C:/Users/bcfri/OneDrive/Desktop/practice code/opencv_tutorial/Faces/train' # theoretical group of folders with faces

# haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = []
# labels= []
# def create_train():
#     for person in people:
#         path = os.path.join(DIR, person)
#         label = people.index(person)

#         for img in os.listdir(path):
#             img_path = os.path.join(path, img)

#             img_array = cv.imread(img_path)
#             gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

#             faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=4)
#             for (x,y,w,h) in faces_rect:
#                 faces_roi =gray[y:y+h, x:x+w]
#                 features.append(faces_roi)
#                 labels.append(label)

# create_train()
# print('Training done ---------------')

# features = np.array(features, dtype='object')
# labels = np.array(labels)

# face_recognizer = cv.face.LBPHFaceRecognizer_create()

# #train the recognizer on the features list and the labels list
# face_recognizer.train(features,labels)

# face_recognizer.save('face_trained.yml')

# np.save('features.npy', features)
# np.save('labels.npy', labels)


############## use trained model

# haar_cascade = cv.CascadeClassifier('haar_face.xml')

# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read('face_trained.yml')

# img = cv.imread('ultimate_group.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# #detect face in image
# faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# for (x,y,w,h) in faces_rect:
#     faces_roi = gray[y:y+h, x:x+w]

#     label,confidence = face_recognizer.predict(faces_roi)
#     print(f'Label = {people[label]} with a confidence of {confidence}')

#     cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
#     cv.rectangle(img,(x,y), (x+w,y+h), (0,255,0), thickness=2)

# cv.imshow('Detected Face', img)

# cv.waitKey(0)

############################################## deep computer vision