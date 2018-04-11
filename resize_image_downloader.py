#download one image and keep a small size
import pandas as pd
import requests as req
from PIL import Image
from io import BytesIO
import os

train_df = pd.read_csv('train.csv')
test_download = train_df.loc[0:10000,]

#suitable download size
TARGET_SIZE = 224
error_download = 0
error_li = []

def create_file(file_name):
    os.mkdir(file_name)
    return None

create_file("""train""")

def image_download(df):
    global error_download
    error_download = 0
    for i in range(len(df)):
        try:
            filename = "./train/{}.jpeg".format(str(df.id[i]))
            print('Detect the image id and generate filename:' + filename)
            path = df.url[i]
            print('Detect the image and download from' + path)
            response = req.get(path)
            pil_image = Image.open(BytesIO(response.content))
            pil_image_rgb = pil_image.convert('RGB')
            pil_image_resize = pil_image_rgb.resize((TARGET_SIZE, TARGET_SIZE))
            pil_image_resize.save(filename)
        except:
            print('Fail to download the image.')
            error_li.append(str(df.url[i]))
            error_download += 1
            pass
    return error_download, error_li



fail = image_download(test_download)
print(fail)

