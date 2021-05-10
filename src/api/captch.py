import cv2
import numpy as np
from matplotlib import pyplot as plt

def boundary(imgpath):
  image = cv2.imread(imgpath)
  # image = cv2.pyrDown(image)
  # image = cv2.pyrDown(image)
  small = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 7))
  
  grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
 
  _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
  connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
  # using RETR_EXTERNAL instead of RETR_CCOMP
  contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # img_ext = cv2.drawContours(image, contours, -1, (0, 255, 0), 10)

  # cv2.imshow('11',connected)
  mask = np.zeros(bw.shape, dtype=np.uint8)
  # print(len(contours))
  # cv2.imshow(contours)

  print(range(len(contours)))
  for idx in range(len(contours)):
      x, y, w, h = cv2.boundingRect(contours[idx])
      mask[y:y+h, x:x+w] = 0
      
      # print("r=", r, "w=",w, "h=",h, "x=",x)
      cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
      r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
      print("r=", r, "w=",w, "h=",h, "x=",x)
      if r > 0.2 and w > 1 and h > 4 : #and x != 0:
      # if x == 0 :
          # cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
          cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255, 0, 0), 1)
  # show image with contours rect
  cv2.imshow('rects', image)
  
  # rgb.save('../resources/testtest.jpg')
  # image.show()
  cv2.waitKey()


#영역별로 자르기
def boundary_sub(imgpath):
  filename = 'mun8'
  imgurl = imgpath + filename + '.jpg'
  image = cv2.imread(imgurl)
  image2 = image.copy()
  # image = cv2.pyrDown(image)
  small = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 7))
  grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
 
  _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
  connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
  # using RETR_EXTERNAL instead of RETR_CCOMP
  contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # img_ext = cv2.drawContours(image, contours, -1, (0, 255, 0), 10)

  mask = np.zeros(bw.shape, dtype=np.uint8)
  # print(len(contours))
  # cv2.imshow('11',connected)

  # x=320; y=150; w=50; h=50;
  # roi = image[y:y+h, x:x+w]  
  # cv2.imshow("img", image)
  # cv2.imshow("1", roi)

  # print(range(len(contours)))
  for idx in range(len(contours)):
      x, y, w, h = cv2.boundingRect(contours[idx])
     
      mask[y:y+h, x:x+w] = 0
      cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
      r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
      
      print("r=", r, "w=",w, "h=",h, "x=",x, idx)
      if r > 0.1 and w > 1 and h > 4 :# and x != 0:
          roi = image[y:y+h, x:x+w] 
          # cv2.imshow('mun8_'+ str(idx) +'_c',roi)
      # if x == 0 :
          # cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
          # cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255, 0, 0), 1)
          # cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255, 0, 0), 1)
          # cv2.rectangle(roi, (x, y), (x+w-1, y+h-1), (255, 0, 0), 1)

          # if w == 194:
          #   cv2.imshow('mun8_'+ str(idx) +'_c', cv2.rectangle(roi, (x, y), (x+w-1, y+h-1), (255, 0, 0), 1))

          cv2.imshow('mun8_'+ str(idx) +'_c1', roi)
          # cv2.imwrite('../resources/opencv/cut2/'+filename+'_'+str(idx)+'_cut2.jpg', roi) #세부 이미지 저장
          cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255, 0, 0), 1) #원본 이미지에 영역 표시하기 위해 

  cv2.imshow('rects', image)
  # cv2.imwrite('../resources/opencv/cut2/'+filename+'_opencv2.jpg', image)  #원본 이미지에 영역 표시된 이미지 저장

  cv2.waitKey()


 

if __name__ == '__main__':
    boundary_sub('../resources/')
    # boundary('../resources/mun7.jpg')

