import argparse
from enum import Enum
import io

from google.cloud import vision
from PIL import Image, ImageDraw

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image

def detect_document(path, feature):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    bounds = []
    boudsWord = []
    # response = client.document_text_detection(image=image)
    

    # print(response.full_text_annotation.pages)
    # for page in response.full_text_annotation.pages:
    for page in document.pages:
    
        for block in page.blocks:
            # print('\nBlock confidence: {}\n'.format(block.confidence))
            for paragraph in block.paragraphs:
                # print('Paragraph confidence: {}'.format(
                #     paragraph.confidence))
                # print(paragraph)
                for word in paragraph.words:
                    # word_text = ''.join([
                    #     symbol.text for symbol in word.symbols
                    # ])
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    # print('word_text', word_text)
                    boudsWord.append(word_text)

                    for symbol in word.symbols:
                      if(feature == FeatureType.SYMBOL):
                        bounds.append(SYMBOL.bounding_box)
                        # print('\tSymbol: {} (confidence: {})'.format(
                        #     symbol.text, symbol.confidence))
                
                    if(feature == FeatureType.WORD):
                      bounds.append(word.bounding_box)
                    # print(word)
                
                if (feature == FeatureType.PARA):
                  bounds.append(paragraph.bounding_box)
                # print(paragraph)
    print(boudsWord)
    return bounds

def render_doc_text(filein, fileout):
  image = Image.open(filein)
  bounds = detect_document(filein, FeatureType.PARA)
  draw_boxes(image, bounds, 'blue') # 문장표시
  # bounds = detect_document(filein, FeatureType.WORD)
  # draw_boxes(image, bounds, 'yellow') # 단어표시
  
  image.show()
  image.save(fileout)

if __name__ == '__main__':
    # detect_text('../resource/mun5.jpg')
    # detect_text1('../resource/mun5.jpg', '../resource/mun5_out2.jpg')
    render_doc_text('../resources/mun5.jpg', '../resources/mun5_out.jpg')
    # detect_document('../resource/mun5.jpg')