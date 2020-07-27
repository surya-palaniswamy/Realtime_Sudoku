#! usr/bin/env python 3

import cv2
import numpy as np
import operator
import pytesseract as pt
# from keras.models import load_model


class reader:

    def __init__(self):
        pass

    def static_reader(self):
        image_path = '/home/a2i2-rl/Desktop/sudoku/image.jpeg'
        image = cv2.imread(image_path)
        return image
    
    def dist(self,A,B):
        x = abs(A[0]-B[0])
        y = abs(A[1]-B[1])
        return (pow((pow(x,2)+pow(y,2)),0.5))

    def process(self,image):
        
        """
        The initial image processing before contouring and fixing the angle of the image
        First the image is converted to grayscale
        Then it is smoothed using a 9x9 Gaussian kernel
        Then use (mean-c) adaptive thresholding for proper light/brightness management profiling of the image and inverted to make the RoI as bright pixel regions
        to dilate the features more for extraction later on.
        """
        # image = self.static_reader()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)


        """Dilating the features, namely the grid lines and the digits."""
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        processed_image = cv2.dilate(thresh, kernel)
        return processed_image


    def get_grid(self):

        """
        Contouring to find the set of connected points, then later selecting the contour with the largest area enclosed to pick out the boundary of the overall sudoku grid
        """
        image = self.static_reader()
        processed_image = self.process(image)
        _, contours, h = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygon = contours[0]

        # image = self.static_reader()
        # show = cv2.drawContours(image,polygon,-1,(0,0,255),3)
        # cv2.imshow('Contours',show)
        # cv2.imshow('image',processed_image)
        # cv2.waitKey(0)

        """
        This is to find the vertices of the polygon, which is the largest contour in our map.
        Since the pixel indexing starts from the top left corner of the image and x being the horizontal axis and y being the vertical axis
        Top-Left = minimum(x+y)
        Top-Right = maximum(x-y) or minimum(y-x)
        Bottom-Right = maximum(x+y)
        Bottom-Left = minimum(x-y) or maximum(y-x)
        """
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        return [polygon[top_left][0],polygon[top_right][0],polygon[bottom_right][0],polygon[bottom_left][0]]

    def transform(self):

        """
        Now we need to get a front perspective of the grid excluding the background and noise.
        This is performed by the 4 point transform of openCV as depicted in code.
        We first select an output image size, in our case we have chosen it to be a square with side equal to the largest side of the grid from original image,
        this makes sure there is no compression of size so there isn't any data loss during the perspective transformation.
        Then using this perspective transformation 3x3 matrix we warp the original image to the desired size and desired perspective.
        """
        coordinates = self.get_grid()
        top_left,top_right,bottom_right,bottom_left = coordinates[0],coordinates[1],coordinates[2],coordinates[3]
        image = self.static_reader()

        #To identify the longest side of the quadrilateral defined by the grid.
        side = max([
            self.dist(bottom_right, top_right),
            self.dist(top_left, bottom_left),
            self.dist(bottom_right, bottom_left),
            self.dist(top_left, top_right)
        ])

        #Assumed original dimensions of the image which needs to be transformed
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
        transform = cv2.getPerspectiveTransform(src, dst)

        # Performs the transformation on the original image
        return cv2.warpPerspective(image, transform, (int(side), int(side)))
    
    def classify(self,cell):
        digit = cell.reshape(1,28,28,1)
        digit = digit/255
        cv2.imshow("Cell",digit)
        cv2.waitKey(0)

    def read_puzzle(self):
        transformed = self.transform()
        cell_size = int(round((transformed.shape[0])/9.0))
        # cell = transformed[cell_size*8:cell_size*9,:cell_size]
        # cv2.imshow("Cell",cell)
        # cv2.waitKey(0)
        processed_image = self.process(transformed)
        filled = []
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                cell = transformed[cell_size*i:cell_size*(i+1),cell_size*j:cell_size*(j+1)]
                # cell_mean = int(cell.mean(axis=0).mean())
                h,w = cell.shape[:2]
                h_buff = int(h/6)
                w_buff = int(w/6)
                cell_mean = int(np.mean([h,w]))
                roi = cell[h_buff:h-h_buff,w_buff:w-w_buff]
                number = pt.image_to_string(cell,config='--psm 6')
                print(number)
                # cv2.imshow('roi',cell)
                # cv2.waitKey(0)



call = reader()
call.get_grid()
call.transform()
call.read_puzzle()
# img = call.static_reader()
# img = img.mean(axis=0)
# print(img)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# height,width = img.shape[:2]
# print(width)
# margin = int(np.mean([height,width])/2.5)
# print(margin)