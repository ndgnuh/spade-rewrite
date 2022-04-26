import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from functools import reduce
from pyvi.ViTokenizer import spacy_tokenize as vi_spacy_tokenize
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)






def Vietocr_img(img,bboxes):
    raw_text=[]
    for box in bboxes:
        img_box=img[box[2]:box[3],box[0]:box[1]]
        img_box=Image.fromarray(img_box)
        text=detector.predict(img_box)
        if text==[]:
            raw_text.append("?")
            continue
        raw_text.append(str(text))
    return raw_text


#Vẽ bbox
def bounding_box(x1,y1,x2,y2,img):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img

#Lấy box từ file txt
def get_bbox(img_name,bbox_dir):
    filename=img_name.replace(".png",".txt")
    with open(os.path.join(bbox_dir, filename)) as f:
        lines=f.readlines()
    bboxes=[]
    for line in lines:
        bboxes.append(eval(line))
    return bboxes

#Tính diện tích dựa vào tập góc
def polygonArea(X, Y, n):
  
    # Initialize area
    area = 0.0
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i   # j is previous vertex to i
    # Return absolute value
    return int(abs(area / 2.0))

#Tạo tập box chữ nhật
def box_convert(bboxes):
    bboxes_new=[]
    for bbox in bboxes:
        x=[]
        y=[]
        #Tạo box chữ nhật và tính diện tích
        for i in range(0,len(bbox)//2):
            x.append(bbox[i*2])
            y.append(bbox[i*2+1])
        bbox_new=[min(x),max(x),min(y),max(y)]
        x1,x2,y1,y2=min(x),max(x),min(y),max(y)

        S_new=polygonArea([x1,x2,x2,x1], [y1,y1,y2,y2],4)
        S_old=polygonArea(x, y, len(x))
        
        #Lọc box nghiêng qua so sánh diện tích
        if abs(S_new-S_old)<0.3*S_new: 
            bboxes_new.append(bbox_new) 
    return bboxes_new

# #EasyOCR
# def EasyOCR(img,bboxes,lang='en'):
#     raw_text=[]
#     for box in bboxes:
#         reader = easyocr.Reader([lang]) # this needs to run only once to load the model into memory
#         result = reader.readtext(img[box[2]:box[3],box[0]:box[1]])
#         if result==[]:
#             raw_text.append("?")
#             continue
#         raw_text.append(str(result[0][1]))
#     return raw_text
    

    
    
#Sắp xếp box theo thứ tự trái -> phải, trên -> dưới
def arrange_bbox(bboxes):
    #x1,x2,y1,y2
    n = len(bboxes)
    xcentres = [(b[0] + b[1]) // 2 for b in bboxes]
    ycentres = [(b[2] + b[3]) // 2 for b in bboxes]
    heights = [abs(b[2] - b[3]) for b in bboxes]
    width = [abs(b[1] - b[0]) for b in bboxes]

    def is_top_to(i, j):
        result = (ycentres[j] - ycentres[i]) > ((heights[i] + heights[j]) / 3)
        return result

    def is_left_to(i, j):
        return (xcentres[i] - xcentres[j]) > ((width[i] + width[j]) / 3)

    # <L-R><T-B>
    # +1: Left/Top
    # -1: Right/Bottom
    g = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(n):
            if is_left_to(i, j):
                g[i, j] += 10
            if is_left_to(j, i):
                g[i, j] -= 10
            if is_top_to(i, j):
                g[i, j] += 1
            if is_top_to(j, i):
                g[i, j] -= 1
    return g


def arrange_row(bboxes=None, g=None, i=None, visited=None):
    if visited is not None and i in visited:
        return []
    if g is None:
        g = arrange_bbox(bboxes)
    if i is None:
        visited = []
        rows = []
        for i in range(g.shape[0]):
            if i not in visited:
                indices = arrange_row(g=g, i=i, visited=visited)
                visited.extend(indices)
                rows.append(indices)
        return rows
    else:
        indices = [j for j in range(g.shape[0]) if j not in visited]
        indices = [j for j in indices if abs(g[i, j]) == 10 or i == j]
        indices = np.array(indices)
        g_ = g[np.ix_(indices, indices)]
        order = np.argsort(np.sum(g_, axis=1))
        indices = indices[order].tolist()
        indices = [int(i) for i in indices]
        return indices

#Show img
def plot_img(img,size):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#Vẽ bounding box
def bounding_box(x1,y1,x2,y2,img):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img

#Sap xep box
def data_arrange(df):
    for i in range(len(df)):
        text_copy=[]
        box_copy=[]
        g=arrange_bbox(df["img_bboxes"][i])
        rows=arrange_row(g=g)
        # # rows
        new_row=reduce(lambda x,y: x+y,rows,[])
        for j in range(len(new_row)):
            box_copy.append(df["img_bboxes"][i][new_row[j]])
            text_copy.append(df["img_texts"][i][new_row[j]])

        df["img_bboxes"][i]=box_copy
        df["img_texts"][i]=text_copy
    return df



def split_bbox(text, bbox):
    # 
    text = re.sub(r"([^0-9]):", r"\1: ", text)
    text = re.sub(r"\s+", " ", text)
    if len(text.strip()) == 0:
        return text, bbox
    
    # DIVIDE THE TEXT INTO MEANINGFUL CHUNKS (NOT VI TOKENIZE STYLE)
    tokens, spaces = vi_spacy_tokenize(text)
    chunks = []
    chunk = ""
    for (token, spacenext) in zip(tokens, spaces):
        chunk += token
        if spacenext:
            chunks.append(chunk)
            chunk = ""
    if len(chunk) > 0:
        chunks.append(chunk)
    
    # FILTER EMPTY
    chunks = [chunk.strip() for chunk in chunks]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]
    
    # DIVIDE THE BOX
    text = ' '.join(chunks)
    new_bboxes = []
    x1, x2, y1, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    for chunk in chunks:
        new_x2 = round(x1 + width * len(chunk) / len(text))
        new_bboxes.append([x1, new_x2, y1, y2])
        x1 = new_x2 + 1
    return chunks, new_bboxes

def split_bboxes(texts, bboxes):
    results = [split_bbox(t, b) for (t, b) in zip(texts, bboxes)]
    new_texts = reduce(lambda x, y: [*x, *y], [r[0] for r in results], [])
    new_bboxes = reduce(lambda x, y: [*x, *y], [r[1] for r in results], [])
    return new_texts, new_bboxes

def split_bboxes_df(df, txt_field, bb_field):
    if isinstance(df, str):
        df = load_csv(df, txt_field, bb_field)
    else:
        df = df.copy()
    for (i, row) in df.iterrows():
        texts = row[txt_field]
        bboxes = row[bb_field]
        texts, bboxes = split_bboxes(texts, bboxes)
        row[txt_field] = texts
        row[bb_field] = bboxes
        df.iloc[i, :] = row
    return df


def load_csv(path, text_field, bbox_field):
    df = pd.read_csv(path)
    df[text_field] = df.apply(lambda r: ast.literal_eval(r[text_field]), axis=1)
    df[bbox_field] = df.apply(lambda r: ast.literal_eval(r[bbox_field]), axis=1)
    return df