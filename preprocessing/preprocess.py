import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import re
import glob


def load_Image(path):
    image=cv2.imread(path)
    return image
def load_Dataset(path):
    images = []
    sorted_files = sorted(glob.glob(path), key=lambda x: int(re.search(r'\d+', x).group()))
    for img in sorted_files:
        n= cv2.imread(img)
        cv2.destroyAllWindows()
        images.append(n)
    print(len(images))
    return images
def threshold_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   # median_filter = cv2.medianBlur(gray_image, 5)
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Get the total number of pixels in the image
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    # Calculate the ratio of black pixels to white pixels
    black_ratio = sum(hist[:50]) / total_pixels
    white_ratio = sum(hist[200:]) / total_pixels
    
    # Set the threshold accordingly
    if black_ratio > white_ratio:
        ret, thresh = cv2.threshold(gray_image,0, 255, cv2.THRESH_BINARY_INV  + cv2.THRESH_OTSU) #50
    else: 
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   #180
    # check salt and pepper to rethreshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    if abs(len(filtered_contours)-len(contours))<8000 :
       return thresh
    else:
        median_filter = cv2.medianBlur(gray_image, 5)
        if black_ratio > white_ratio:
            ret, thresh = cv2.threshold(median_filter,0, 255, cv2.THRESH_BINARY_INV  + cv2.THRESH_OTSU) #50
        else: 
            ret, thresh = cv2.threshold(median_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   #180
    return thresh
def assure_white_bg(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    total_pixels = image.shape[0] * image.shape[1]
    # Calculate the ratio of black pixels to white pixels
    black_ratio = sum(hist[:50]) / total_pixels
    white_ratio = sum(hist[200:]) / total_pixels
    thresh=image.copy()
    if black_ratio < white_ratio:
        #ret, thresh = cv2.threshold(image,50, 255, cv2.THRESH_BINARY_INV)
        thresh=255-image
    return thresh
def get_words(image):
    # Get the contours of the image
    process_on=image.copy()
    dilated_image = cv2.dilate(process_on, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out the small contours
    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    # Sort the contours from left to right
    filtered_contours = sorted(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    # cv2.drawContours(x, filtered_contours, -1, (0, 255, 0), 3)
    # cv2.imshow('contours', x)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cropped_images = []
    
    # Iterate through each contour and crop the image
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w*h >1000 ):
            cropped_image = image[y:y+h, x:x+w]
            cropped_images.append(cropped_image)
    
    # Display the cropped images (optional)
    # for i, cropped_img in enumerate(cropped_images):
    #     cv2.imshow(f'Cropped Image {i}', cropped_img)
    #     cv2.waitKey(0)
    
    # cv2.destroyAllWindows()
    #print (f'Number of words: {len(cropped_images)}')
    return cropped_images
def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)