from lenses import lens
from .box import Box
from .models import SpadeData

def from_google(response):
    def box_from_google_api_poly(poly):
        #   [{'x': 25, 'y': 446},
        #    {'x': 549, 'y': 446},
        #    {'x': 549, 'y': 1234},
        #    {'x': 25, 'y': 1234}]
        x = [p['x'] for p in poly]
        y = [p['y'] for p in poly]
        return Box(x + y, Box.Type.X4Y4)

    boxes_focus = lens['textAnnotations'].Each()['boundingPoly']['vertices'].F(
        box_from_google_api_poly).collect()
    texts_focus = lens['textAnnotations'].Each()['description'].collect()

    boxes = boxes_focus(response)
    x0, y0, w, h = boxes[0].xywh

    def normalize(b):
        x = [max(x - x0, 0) for x in b.x4y4[:4]]
        y = [max(y - y0, 0) for y in b.x4y4[4:]]
        return Box(x + y, Box.Type.X4Y4)
    boxes = [normalize(b) for b in boxes]

    return SpadeData(texts=texts_focus(response)[1:],
                     boxes=boxes[1:],
                     width=w,
                     height=h,
                     relations=None)

def from_doctr(bboxes,raw_text,h,w): #and vietocr
    new_boxes=[Box(box, Box.Type.XXYY) for box in bboxes]
    return SpadeData(texts=raw_text,
                     boxes=new_boxes,
                     width=w,
                     height=h,
                     relations=None)

