import cv2
import numpy as np

def covariance(img1, img2):
    xmean = np.mean(img1)
    ymean = np.mean(img2)
    M, N = img1.shape
    sum_ = 0
    for i in range(M):
        for j in range(N):
            sum_ += (img1[i][j] - xmean)*(img2[i][j] - ymean)

    return sum_/((M*N) - 1)

def loss_correl(img1, img2):
    cov = covariance(img1, img2)
    a = cov/(covariance(img1,img1) * covariance(img2,img2))
    return a

def lum_distortion(img1, img2):
    xmean = np.mean(img1)
    ymean = np.mean(img2)
    top = 2*xmean*ymean
    bot = (xmean**2) + (ymean**2)
    return top/bot

def contrast_distortion(img1,img2):
    xvar = covariance(img1, img1)
    yvar = covariance(img2, img2)
    top = xvar * yvar * 2
    bot = (xvar**2) + (yvar**2)
    return top/bot


def Q(img1, img2):
    lc = loss_correl(img1, img2)
    ld = lum_distortion(img1, img2)
    cd = contrast_distortion(img1, img2)
    return lc*ld*cd

def Q2(img1, img2):
    top = 4*covariance(img1,img2)*np.mean(img1)*np.mean(img2)
    bot = (covariance(img1, img1) + covariance(img2,img2)) * ((np.mean(img1)**2) + (np.mean(img2)**2))
    return top/bot
def mse(img1, img2):
    M, N = img1.shape
    sum_ = 0
    for i in range(M):
        for j in range(N):
            sum_ += (img1[i][j] - img2[i][j])**2

    return sum_/(M*N)

img = cv2.imread('imgs/Image42.png', 0)
blur = cv2.imread('imgs/blur.png', 0)
print(Q2(img, img))
