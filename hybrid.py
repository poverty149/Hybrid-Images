from align import align
from filters import low_pass,high_pass
import cv2
import matplotlib.pyplot as plt
def hybrid(img1,img2,sigma,size):
    # img1_aligned=align(img1,img2)
    lp_img1=low_pass(img1,sigma,size)
    hp_img2=high_pass(img2,sigma,size)
    return hp_img2+lp_img1

img1=cv2.imread("cat.bmp")
img1_gray=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

img2=cv2.imread("dog.bmp")
img2_gray=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
res=hybrid(img2_gray,img1_gray,5,(20,20))

plt.figure()
# plt.subplot(211)
# plt.imshow(img1,cmap='gray')
# plt.subplot(212)
plt.imshow(res*255,cmap="gray")
plt.show()