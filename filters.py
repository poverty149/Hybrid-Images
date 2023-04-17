from scipy import fftpack
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import imageio

def convolve(img,kernel):
    m,n=kernel.shape
    res=np.zeros(img.shape)
    if img.ndim==3:
        h,w,c=img.shape
    else:
        c=1
        h,w=img.shape
        img=np.reshape(img,(h,w,1))
        
    kernel=np.ravel(kernel)
    new_img=np.zeros((m+h-1,n+w-1,c))
    new_img[m//2:m//2+h,n//2:n//2+w]=img
    
    for x in range(w):
        for y in range(h):
            temp=np.reshape(new_img[y:y+m,x:x+n],(m*n,c))
            
            res[y,x]=np.dot(kernel,temp)
    return res
def gaussian_filter(sigma,size):
    m,n=size
    gaussian=np.zeros((m,n))
    for x in range(-m//2,m//2):
        for y in range(-n//2,n//2):
            c=1/(2.0*np.pi*sigma**2.0)
            exp_term=np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m//2,y+n//2]=exp_term*c

    return gaussian


# def low_pass():
#     img = imageio.imread('dog.bmp',as_gray=True)
#     fft = fftpack.fftshift(fftpack.fft2(img))

# #Create a low pass filter image
#     x,y = img.shape[0],img.shape[1]
#     #size of circle
#     e_x,e_y=50,50
#     #create a box 
#     bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

#     low_pass=Image.new("L",(img.shape[0],img.shape[1]),color=0)

#     # draw1=ImageDraw.Draw(low_pass)
#     # draw1.ellipse(bbox, fill=1)

#     low_pass_np=np.array(low_pass)

#     #multiply both the images
#     filtered=np.multiply(fft,low_pass_np)

#     #inverse fft
#     ifft= np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
#     ifft= np.maximum(0, np.minimum(ifft, 255))
#     imageio.imsave('fft-then-ifft.png', ifft.astype(np .uint8))
#     return ifft

def low_pass(img,sigma,size):
    return convolve(img,gaussian_filter(sigma,size))
def high_pass(img,sigma,size):
    return img-low_pass(img,sigma,size)

import numpy as np
import cv2
from scipy import signal


def high_pass_function(image):

    # edge detection filter
    kernel = np.array([[-1.0, -1.0, -1.0],
                       [-1.0, 8.0, -1.0],
                       [-1.0, -1.0, -1.0]])

    # kernel = np.array([[-1, 0, 1],
    #                    [-2, 0, 2],
    #                    [-1, 0, 1]])
    #
    # kernel = np.array([[-1, -2, -1],
    #                    [0, 0, 0],
    #                    [1, 2, 1]])

    #
    # # # Sharpen
    # kernel = np.array([[0,  0,  -1, 0,  0],
    #                    [0,  -1, -2, -1, 0],
    #                    [-1, -2, 16, -2, -1],
    #                    [0,  -1, -2, -1, 0],
    #                    [0,  0,  -1, 0,  0]
    #                    ])

    # # Emboss
    # kernel = np.array([[2, 1, 0],
    #                    [1, 0, -1],
    #                    [0, -1, -2]])

    # # Sharpen
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 4, -1],
    #                    [0, -1, 0]])

    convolved_image = signal.convolve2d(image, kernel)

    truncated_image = truncate_v2(convolved_image, kernel)

    high_pass_filtered_image = (truncated_image + image)

    return high_pass_filtered_image


def low_pass_function(image):

    # # Gaussian Filter (Smoothing)
    # kernel = np.array([[1, 2, 1],
    #                    [2, 4, 2],
    #                    [1, 2, 1]]) / 16

    # low pass filter
    kernel = np.ones((10, 10)) / 100

    convolved_image = signal.convolve2d(image, kernel)

    truncated_image = truncate_v2(convolved_image, kernel)

    low_pass_filtered_image = truncated_image

    return low_pass_filtered_image
def truncate_v2(image, kernel):

    m, n = kernel.shape
    m = int((m-1) / 2)

    for i in range(0, m):
        line, row = image.shape
        image = np.delete(image, line-1, 0)
        image = np.delete(image, row-1, 1)
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
    return image
# img1=cv2.imread('cat.bmp')
# img2=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
# lp=high_pass(img2,5,(80,80))
# # res=img2-lp
# plt.figure()
# plt.subplot(211)
# plt.imshow(img2,cmap='gray')
# plt.subplot(212)
# plt.imshow(lp,cmap="gray")
# plt.show()
