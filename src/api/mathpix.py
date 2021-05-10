import sys
import base64
import requests
import json
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append('../../config/config.py')
import json
with open('../../config/config.json') as f:
    config = json.load(f)

# def render_doc_text(filein, fileout):
def render_doc_text():
    # image = Image.open(filein)
    fileurl = '../resources/opencv/cut2/'
    # print(filein)
    # filename = ['1','2','3','4','6','5','7','8','9','10','11','12','13','14','15','16','17','18','19']
    filename = ['1','2','5','7','9','10','11','12','13','14','15','16']
    for i in filename :
        filename11 = 'mun7_' + i + '_cut2.jpg'
    # filename11 = 'mun5_8_ys.jpg'
        print(filename11)
        # file_path = filein
        file_path = fileurl+filename11
        print(file_path)
        image_url = "data:image/jpg;base64," + base64.b64encode(open(file_path, "rb").read()).decode()
        r = requests.post("https://api.mathpix.com/v3/text",
            data = json.dumps({'src': image_url}),
            headers = {"app_id": config["mathPix"]["app_id"], "app_key" : config["mathPix"]["app_key"], "Content-Type": "application/json"},)
    # print(json.dumps(json.loads(r.text), indent=4, sort_keys=True))


    # mathtext = json.dumps(json.loads(r.text))

    # print(mathtext)
    # print(json.dumps(json.loads(r.data), indent=4, sort_keys=True))
    # image.shodw()

def render_one():
    fileurl = '../resources/opencv/cut2/'
    filename11 = 'mun7_12_cut2_3.jpg'

    print(filename11)
    # file_path = filein
    file_path = fileurl+filename11
    print(file_path)

    image = cv2.imread(file_path)


    # image = cv2.resize(image, dsize=(0, 0), fx=2, fy=5, interpolation=cv2.INTER_LINEAR)
    height, width = image.shape[:2]

    image = cv2.resize(image, (int(width*0.8), int(height*0.8)), \
                         interpolation=cv2.INTER_AREA)
    # cv2.imshow('11', image)
    # cv2.waitKey(0)
    image_url = "data:image/jpg;base64," + base64.b64encode(open(file_path, "rb").read()).decode()
    r = requests.post("https://api.mathpix.com/v3/text",
        data = json.dumps({'src': image_url}),
        headers = {"app_id": config["mathPix"]["app_id"], "app_key" : config["mathPix"]["app_key"], "Content-Type": "application/json"},)

def shape():
    #mun5_14_cut2
    image = cv2.imread('../resources/opencv/cut2/mun7_10_cut2_shape.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = image.copy()
    
    image = cv2.resize(image, dsize=(0, 0), fx=2, fy=5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('ori', image)

    # 커널 생성(대상이 있는 픽셀을 강조)
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

    # 커널 적용 
    image_sharp = cv2.filter2D(image, -1, kernel)

    # fig, ax = plt.subplots(1,2, figsize=(10,5))
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title('Original Image')
    # ax[1].imshow(image_sharp, cmap='gray')
    # ax[1].set_title('Sharp Image')

    # cv2.imshow('11', image_sharp)
    # cv2.imwrite('../resources/opencv/cut2/mun7_10_cut2_shape.jpg', image_sharp)  
    # cv2.waitKey()

    # img = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    # cv2.imshow("1111", img)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 열림 연산 적용 ---②
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, k)
    # opening2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    # # 닫힘 연산 적용 ---③
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)
    # closing2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)

    # erosion = cv2.erode(image, k)
    # dst = cv2.dilate(image2, k)
    image_sharp = cv2.filter2D(opening, -1, kernel)


    merged = np.hstack((image, opening))
    merged1 = np.hstack((image, image_sharp))
    merged2 = np.hstack((image, closing))
    # merged2_1 = np.hstack((img, opening2))
    # merged3 = np.vstack((merged, merged2))

    cv2.imshow('erode', merged)
    cv2.imshow('er22ode', merged1)
    
    cv2.imshow('er22o123123de', merged2)
    # cv2.imwrite('../resources/opencv/cut2/mun5_14_cut2_opneing_big_c.jpg',erosion)
    # cv2.imshow('1', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def shape2():    
    image = cv2.imread('../resources/opencv/cut2/mun7_10_cut2_shape.jpg')
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # dst = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    dst2 = cv2.resize(image, dsize=(0, 0), fx=2, fy=5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('11',dst2)
    # cv2.imwrite('../resources/opencv/cut2/mun7_10_cut2_shape_color.jpg', final)  
    cv2.waitKey(0)

def removeLine():
    image = cv2.imread('../resources/opencv/cut2/mun7_7_cut2.jpg')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    cv2.imshow('thresh', thresh)
    cv2.imshow('detected_lines', detected_lines)
    cv2.imshow('image', image)
    cv2.imshow('result', result)
    cv2.waitKey()  

if __name__ == '__main__':
    # render_doc_text('../resources/mun5.jpg', '../resources/opencv/mun5_1_out.jpg')
    # render_doc_text()
    # shape()
    render_one()
    # shape2()
    # removeLine()
