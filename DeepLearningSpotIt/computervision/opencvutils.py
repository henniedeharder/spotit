import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
from os.path import join

class Image:
    def __init__(self, path_to_img):
        '''Uses path to image to process an image.

        :param path_to_img: path to the image

        :ivar path: path to the image
        :ivar image_name: name of image without directories (with extension) 
        :ivar image: raw image to use for processing'''

        self.path = path_to_img
        self.image_name = self._image_name()
        self.wo_extension, self.extension = self._image_name_ext()
        self.image = self._read_image()

        self.cnts_images = []
        self.cntsx = []
        self.cntsy = []
        self.predictions = dict()

    # instance methods
    def _image_name(self):
        whole = self.path.split('\\')[-1]
        return whole

    def _image_name_ext(self):
        wo_extension, extension = self.image_name.split('.')[0], self.image_name.split('.')[1]
        extension = f'.{extension}'
        return wo_extension, extension

    def _read_image(self):
        return cv2.imread(self.path)

    def save_image(self, directory, image, addition=''):
        '''Save image in specified directory
        
        :param directory: directory to save the image
        :param image: image you want to save'''

        cv2.imwrite(join(directory, f'{self.wo_extension}{addition}{self.extension}'), image)

    def contrast_resized_blurred(self):
        '''Apply contrast, resize and blur to the original image'''
        return Image.combine(self.image)

    # static methods
    def add_contrast(image):
        '''Add contrast to an image
        
        :param image: image to apply the contrast to
        :return: image with contrast'''
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def resize_image(image, size=(800, 800)):
        '''Resize image
        
        :param image: image to resize
        :param size: tuple with new size
        :return: resized image'''
        return cv2.resize(image, size)

    def blur_image(image):
        '''Blur image
        
        :param image: image to blur
        :return: blurred image'''
        return cv2.GaussianBlur(image, (11, 11), 0)

    def combine(image, contrast=True, resize=True, blur=False):
        if contrast:
            img = Image.add_contrast(image)
        if resize:
            img = Image.resize_image(img)
        if blur:
            img = Image.blur_image(img)
        return img 

    def show_image(image):
        plt.imshow(image)
        plt.show()

    def gray_image(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def convert_color(image, toRGB=True):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def thresh_image(image, threshold=190):
        return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

    def grab_contours_by_area(image, original=True, threshold=190, reverse=False, all=False, area=False):
        '''Grab contours from image
        
        :param image: image to grab contours from
        :param original: if it's a threshold image you can set original to False, default True
        :param threshold: threshold for threshold image, default 190
        :param reverse: set to True if you want to reverse the threshold image, default False
        :param all: if False, it only returns outer contours, default False
        :param area: only keep items with area greater than this value

        :return: contours sorted by area starting with largest'''
        if original:
            gray = Image.gray_image(image)
            threshold = Image.thresh_image(gray, threshold=threshold)
            if reverse:
                threshold = cv2.bitwise_not(threshold)
        if all:
            cnts = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else: 
            cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if area:
            cnts = [c for c in cnts if cv2.contourArea(c) > area]
        return sorted(cnts, key=cv2.contourArea, reverse=True)

    def draw_contour(image, cnt, rect=False):
        '''Draw contour on image
        
        :param image: image where you want the contour visualized
        :param cnt: contour you want to draw
        :param rect: set to True if you want a bounding rectangle, default False
        
        :return: image with contour drawn on it'''
        if rect:
            x,y,w,h = Image.get_rect_coordinates_around_contour(cnt)
            contour_image = cv2.rectangle(image, (x, y), (x+w, y+h), color = (255, 0, 0), thickness = 2)
        else:
            contour_image = cv2.drawContours(image, [cnt], -1, (255, 0, 0), 3)
        return contour_image

    def add_text(image, text, x=10, y=10, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
        return cv2.putText(image, text, (x,y), fontFace=font, fontScale=1, color=(255,0,0), thickness=thickness)

    def keep_contour(image, cnt):
        gray = Image.gray_image(image)
        mask = np.zeros(gray.shape, np.uint8)
        mask = cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        output = image.copy()
        return cv2.bitwise_and(output, output, mask=mask)

    def keep_contour_with_white_background(image, cnt):
        gray = Image.gray_image(image)
        mask = np.zeros(gray.shape, np.uint8)
        mask = cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        bk = np.full(image.shape, 255, dtype=np.uint8)
        fg_masked = cv2.bitwise_and(image, image, mask=mask)
        mask = cv2.bitwise_not(mask)
        bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
        mask = cv2.bitwise_not(mask)
        return cv2.bitwise_or(fg_masked, bk_masked)

    def get_rect_coordinates_around_contour(cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        return x,y,w,h

    def bounding_square_around_contour(cnt):
        x,y,w,h = Image.get_rect_coordinates_around_contour(cnt)
        # create squares io rects
        if w < h:
            x += int((w-h)/2)
            w = h
        else:
            y += int((h-w)/2)
            h = w
        return x, y, w, h

    def take_out_roi(image, x, y, w, h):
        return image[y:y+h, x:x+w]

    def add_2_images(image1, image2, hor=True):
        if hor:
            img = np.concatenate((image1, image2), axis=1)
        else:
            img = np.concatenate((image1, image2), axis=0)
        return img

    def save_image_(directory, image, name, addition=''):
        '''Save image in specified directory
        
        :param directory: directory to save the image
        :param image: image you want to save'''

        cv2.imwrite(join(directory, f'{name}.jpg'), image)