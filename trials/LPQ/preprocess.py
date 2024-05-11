import numpy as np
import cv2


def preprocess_image(img, desired_size=(256, 256)):

    # Apply median filter to remove salt and pepper noise
    denoised_img = cv2.medianBlur(img, 3)

    # Threshold the image using Otsu's method
    _, thresh_img = cv2.threshold(
        denoised_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Check the color of the background at all four corners
    corners = [thresh_img[0, 0], thresh_img[0, -1],
               thresh_img[-1, 0], thresh_img[-1, -1]]
    white_corners = np.sum(np.array(corners) == 255)

    # If the majority of the corners are white, invert the image to make the background black
    if white_corners > 2:
        thresh_img = cv2.bitwise_not(thresh_img)

    # Resize the image to the desired size
    resized_img = cv2.resize(thresh_img, desired_size,
                             interpolation=cv2.INTER_AREA)

    return resized_img
