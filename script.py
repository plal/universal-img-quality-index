import cv2
import numpy as np
import random

def sp_noise(image,prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

'''def gauss_noise(image):
    row,col = image.shape
    mean = 0
    gauss = np.random.normal(mean,1,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy
'''

'''def gauss_noise(image,prob):
    img_ = image.copy()
    noise = np.random.normal(0,1,[img_.shape[0],img_.shape[1]])
    print(noise)
    noisy_img = img_+noise
    for i in range(img_.shape[0]):
        for j in range(img_.shape[1]):
            if noisy_img[i][j] > 1:
                noisy_img[i][j] = 1
            elif noisy_img[i][j] < 0:
                noisy_img[i][j] = 0
    for i in range(img_.shape[0]):
        for j in range(img_.shape[1]):
            noisy_img[i][j] = noisy_img[i][j]*3
    return noisy_img
'''

def gauss(image):
    rows,cols = image.shape
    mean = 1
    gauss = np.random.normal(mean,1,(rows,cols))
    gauss = gauss.reshape(rows,cols)
    noisy = image+gauss
    return noisy


img = cv2.imread('imgs/Image42.png',0)
#print(type(img))
gauss_img = gauss(img)
#print(type(gauss_img))
#cv2.imwrite('imgs/gauss.png',gauss_img)
blur = cv2.blur(img,(5,5))
cv2.imwrite('imgs/blur.png',blur)
sp_img = sp_noise(img,0.004)
cv2.imwrite('imgs/sp.png',sp_img)
