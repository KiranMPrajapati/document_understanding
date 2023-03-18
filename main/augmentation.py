import cv2 
import numpy as np 

def scale_mat(height, width, scale_factor):

    centerX = (width) / 2
    centerY = (height) / 2

    tx = centerX - centerX * scale_factor
    ty = centerY - centerY * scale_factor

    scale_mat = np.array([[scale_factor, 0, tx], [0, scale_factor, ty], [0., 0., 1.]])

    return scale_mat

def rotate_mat(height, width, angle):

    centerX = (width - 1) / 2
    centerY = (height - 1) / 2

    rotation_mat = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
    rotation_mat = np.vstack([rotation_mat, [0., 0., 1.]])

    return rotation_mat

def translate_mat(horizontal, vertical):

    translation_mat = np.array([[1., 0, horizontal], [0, 1., vertical], [0.0, 0.0, 1.0]])
    return translation_mat

def shear_mat(height, width, shear_factor):

    shear_mat = np.float32([[1, 0, 0], [shear_factor, 1, 0], [0, 0, 1]])
    shear_mat[0, 2] = -shear_mat[0, 1] * height/2
    shear_mat[1, 2] = -shear_mat[1, 0] * width/2

    return shear_mat

def elation_mat(height, width, elation_x, elation_y):

    elation_mat = np.array([[1., 0., 0.], [0., 1., 0.], [elation_x, elation_y, 1.]])

    elation_mat[0, 0] = 1 + elation_x * width/2
    elation_mat[0, 1] = elation_y * width/2
    elation_mat[0, 2] = (-width/2) * (1 + elation_x * width/2) - elation_y * height * width / 4 + width/2

    elation_mat[1, 0] = elation_x * height/2
    elation_mat[1, 1] = 1 + elation_y * height/2
    elation_mat[1, 2] = -elation_x * width * height/4 - (height/2)*(1 + elation_y*height/2) + height/2

    elation_mat[2, 2] = -elation_x*width/2 - elation_y*height/2 + 1

    return elation_mat 