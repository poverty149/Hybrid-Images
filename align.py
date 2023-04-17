import cv2
import numpy as np
def _extract_keypoints_sift(img1,img2):
        """extract keypoints and descriptors from both images"""
        ## cv2.xfeatures2d.SIFT_create() #for older versions
        
        sift = cv2.SIFT_create() 
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        # knn search and ration test
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2, k=2)
        match_pts1, match_pts2 = _filterMatches_ratio_test(matches, kp1, kp2)
        return match_pts1,match_pts2
def _filterMatches_ratio_test( matches, kp1, kp2):
        """ Filters sift feature matches based on Ratio test """
        good = []
        pts1, pts2 = [], []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append([1,0])
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)   
            else:
                good.append([0,0]) 
        pts1, pts2 = np.int32(pts1), np.int32(pts2)
        good=good
        return pts1, pts2
def align(img_src, img_des):
      pts1,pts2=_extract_keypoints_sift(img_src,img_des)
      H,mask=cv2.findHomography(pts1,pts2)
      imw, imh = img_des.shape[1], img_des.shape[0]
      im_warped=cv2.warpPerspective(img_src,H,(imw,imh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT)
      return im_warped

img1=cv2.imread("cat.bmp")
img1_gray=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

img2=cv2.imread("dog.bmp")
img2_gray=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
align(img1,img2)