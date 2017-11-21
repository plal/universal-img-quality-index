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

def gauss(image,var):
    row,col = image.shape
    mean = 0
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image+gauss
    return noisy


def speckle(image,var):
    row,col = image.shape
    mean = 0
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + image*gauss
    return noisy

def contrast_stretching(image):
    output = np.zeros(image.shape, np.uint8)
    max_int = image.max()
    min_int = image.min()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j] = ((image[i][j]-min_int)/(max_int - min_int))*255
    return output

'''Goldhill image and its distortions'''
goldhill = cv2.imread('imgs/goldhill/goldhill.png',0)
#jpg compression
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 7]
result, encimg = cv2.imencode('.jpg', goldhill, encode_param)
goldhill_jpg = cv2.imdecode(encimg, 0)
cv2.imwrite('imgs/goldhill/goldhill_jpg.png',goldhill_jpg)
#gaussian noise
goldhill_gaussian = gauss(goldhill,100)
cv2.imwrite('imgs/goldhill/goldhill_gaussian.png',goldhill_gaussian)
#contrast stretching
goldhill_cs = contrast_stretching(goldhill)
cv2.imwrite('imgs/goldhill/goldhill_cs.png',goldhill_cs)
#multiplicative speckle
goldhill_speckle = speckle(goldhill,0.01)
cv2.imwrite('imgs/goldhill/goldhill_speckle.png',goldhill_speckle)
#blur
goldhill_blur = cv2.blur(goldhill,(6,6))
cv2.imwrite('imgs/goldhill/goldhill_blur.png',goldhill_blur)
#salt and pepper
goldhill_sp = sp_noise(goldhill,0.004)
cv2.imwrite('imgs/goldhill/goldhill_sp.png',goldhill_sp)

'''Lena image and its distortions'''
lena = cv2.imread('imgs/lena/lena.png',0)
#jpg compression
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 3]
result, encimg = cv2.imencode('.jpg', lena, encode_param)
lena_jpg = cv2.imdecode(encimg, 0)
cv2.imwrite('imgs/lena/lena_jpg.png',lena_jpg)
#gaussian noise
lena_gaussian = gauss(lena,250)
cv2.imwrite('imgs/lena/lena_gaussian.png',lena_gaussian)
#contrast stretching
lena_cs = contrast_stretching(lena)
cv2.imwrite('imgs/lena/lena_cs.png',lena_cs)
#multiplicative speckle
lena_speckle = speckle(lena,0.01)
cv2.imwrite('imgs/lena/lena_speckle.png',lena_speckle)
#blur
lena_blur = cv2.blur(lena,(12,12))
cv2.imwrite('imgs/lena/lena_blur.png',lena_blur)
#salt and pepper
lena_sp = sp_noise(lena,0.007)
cv2.imwrite('imgs/lena/lena_sp.png',lena_sp)

'''Couple image and its distortions'''
couple = cv2.imread('imgs/couple/couple.png',0)
#jpg compression
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
result, encimg = cv2.imencode('.jpg', couple, encode_param)
couple_jpg = cv2.imdecode(encimg, 0)
cv2.imwrite('imgs/couple/couple_jpg.png',couple_jpg)
#gaussian noise
couple_gaussian = gauss(couple,100)
cv2.imwrite('imgs/couple/couple_gaussian.png',couple_gaussian)
#contrast stretching
couple_cs = contrast_stretching(couple)
cv2.imwrite('imgs/couple/couple_cs.png',couple_cs)
#multiplicative speckle
couple_speckle = speckle(couple,0.004)
cv2.imwrite('imgs/couple/couple_speckle.png',couple_speckle)
#blur
couple_blur = cv2.blur(couple,(3,3))
cv2.imwrite('imgs/couple/couple_blur.png',couple_blur)
#salt and pepper
couple_sp = sp_noise(couple,0.0025)
cv2.imwrite('imgs/couple/couple_sp.png',couple_sp)
