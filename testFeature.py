import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

path = "/Users/jin/Q_Mac/Local_Data/02_03_2022/costco/2022-02-03T10-48-16/"
index  = 6099
img = cv.imread(path+"color/"+str(index) +".png")
mask = cv.imread(path+"people_mask/"+str(index) +".png",0)
mask = cv.bitwise_not(mask)
kernel = np.ones((7,7), np.uint8)
# mask = cv.dilate(mask, kernel, iterations=15)

mask = cv.erode(mask, kernel, iterations=2)

 

res = cv.bitwise_and(img,img,mask = mask)

plt.imshow(res),plt.show()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,150,0.01,25,mask = mask)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()


img = cv.imread(path+"color/"+str(index) +".png")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,150,0.01,25)
corners = np.int0(corners)

res = cv.bitwise_and(img,img,mask = mask)

for i in corners:
    x,y = i.ravel()
    cv.circle(res,(x,y),3,(0, 255, 0),-1)


plt.imshow(res),plt.show()

cv.imwrite("/Users/jin/Third_Party_Packages/yolact_cpu/maskfeature.png",res)