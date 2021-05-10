import cv2
import numpy as np

#3. Blur 처리: 이미지의 노이즈를 제거하는 작업

# 평균 블러링(Average Blurring) 
# 모든 픽셀에 동일한 가중치 부여 
def avg_blur(image, kernel_size=(5,5)):
  image = cv2.imread(image)
  cv2.imshow('11', image)
  cv2.waitKey(0)
  # return cv2.blur(image, kernel_size) 

# 가우시안 블러링(Gaussian Blurring) 
# 중심에 있는 픽셀에 높은 가중치 부여 
def gau_blur(image, kernel_size=(5,5)):
  return cv2.GaussianBlur(image, kernel_size, 0) 
  
# 미디언 블러링(Median Blurring) 
# 커널 크기 내 픽셀을 크기 순으로 정렬 후 중간값을 뽑아 픽셀로 사용 
def median_blur(image, kernel_size=5): 
  return cv2.medianBlur(image, ksize=kernel_size)


# 4. Morphology 변환: 이미지의 특정 부분을 조건에 맞춰 변환하는 작업
# 모폴로지 변환(Morphology transformation)은 특정 부분을 단순화하거나 제거, 보정하는 것을 의미합니다. 보통 binary 이미지나 grayscale 이미지에서 흰색으로 표현된 객체의 형태를 개선하기 위해서 사용된다고 하네요. 


# Dilation(팽창) 
# 객체 외곽 픽셀 주변에 1(흰색) 추가 -> 이미지 경계를 기준으로 팽창하는 효과 발생 
def delation(image): 
  kernel = np.ones((5,5), np.unit8) 
  result = cv2.dilate(image, kernel, iterations=1) 
  return result 

# Erosion(침식) 
# 객체 외곽 픽셀 주변에 0(검은색) 추가 -> 이미지 경계를 기준으로 침식하는 효과 발생 
def erosion(image): 
  kernel = np.ones((5,5), np.uint8) 
  result = cv2.erode(image, kernel, iterations=1) 
  return result 

# Opening(침식 -> 팽창) 
# 이미지 상의 작은 잡티, 물체 등을 제거하는 효과 발생 
def open(image):
  image = cv2.imread(image)
  image2 = image.copy()
  
  kernel = np.ones((1,2), np.uint8) 
  result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) 
  cv2.imshow('11', result)
  cv2.imshow('22', image2)
  cv2.waitKey(0)
  # return result 

# Closing(팽창 -> 침식) 
# 전체적인 윤곽을 뚜렷하게 보이는 효과 발생 
def close(image): 
  image = cv2.imread(image)
  image2 = image.copy() 

  image = cv2.resize(image, dsize=(0, 0), fx=2, fy=5, interpolation=cv2.INTER_LINEAR)

  kernel = np.ones((1,2), np.uint8) 
  
  result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) 
  result2 = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel) 
  result3 = cv2.erode(result, kernel, iterations=1) 
  
  cv2.imshow('11', result3)
  cv2.imshow('22', result2)
  
  cv2.waitKey(0)



def removeline(image):
  image = cv2.imread(image)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  kernel = np.array([[0, 0, 1],
                    [1, 1, 1],
                    [0, 0, 0]], dtype=np.uint8)
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

  cv2.imshow('result', opening)  
  cv2.waitKey(0)
  

if __name__ == '__main__':
  image = '../resources/opencv/cut2/mun7_7_cut2.jpg'
  # avg_blur(image)
    # gau_blur(image)
    # median_blur(image)

    # delation(image)
    # erosion(image)
  # open(image)
  # close(image)
  removeline(image)