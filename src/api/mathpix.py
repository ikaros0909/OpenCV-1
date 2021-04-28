import sys
import base64
import requests
import json
from PIL import Image, ImageDraw
import cv2
import numpy as np
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

if __name__ == '__main__':
    # render_doc_text('../resources/mun5.jpg', '../resources/opencv/mun5_1_out.jpg')
    render_doc_text()
