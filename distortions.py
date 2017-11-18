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


   def gauss_noise(image,prob):
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


   def gauss(image):
    row,col = image.shape
    mean = 1
    gauss = np.random.normal(mean,1,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image+gauss
    return noisy


   def speckle(image):
    row,col = image.shape
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)
    noisy = image + image*gauss
    return noisy
'''

def contrast_stretching(image):
    output = np.zeros(image.shape, np.uint8)
    max_int = image.max()
    min_int = image.min()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j] = ((image[i][j]-min_int)/(max_int - min_int))*255
    return output


img = cv2.imread('imgs/Image6.png',0)
#encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
#result, encimg = cv2.imencode('.jpg', img, encode_param)
#cv2.imwrite('imgs/jpeg_comp.png',encimg)
#gauss_img = gauss(img)
#cv2.imwrite('imgs/gauss.png',gauss_img)
cs_image = contrast_stretching(img)
cv2.imwrite('imgs/cs2.png',cs_image)
#speckle_img = speckle(img)
#cv2.imwrite('imgs/speckle.png',speckle_img)
blur = cv2.blur(img,(5,5))
cv2.imwrite('imgs/blur2.png',blur)
sp_img = sp_noise(img,0.004)
cv2.imwrite('imgs/sp2.png',sp_img)
