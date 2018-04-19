#download one image and keep a small size
import pandas as pd
import requests as req
from PIL import Image
from io import BytesIO
import os

def create_file(file_name):
    os.mkdir(file_name)
    return None

def image_download(df):
    global error_download, sample_count, error_li
    TARGET_SIZE = 224
    error_download = 0
    error_li = []
    sample_count = 0
    for i in range(len(df)):
        try:
            filename = "./train/{}.jpeg".format(str(df.id.iloc[i]))
            print('Detect the image id and generate filename:' + filename)
            path = df.url.iloc[i]
            print('Detect the image and download from' + path)
            response = req.get(path)
            pil_image = Image.open(BytesIO(response.content))
            pil_image_rgb = pil_image.convert('RGB')
            pil_image_resize = pil_image_rgb.resize((TARGET_SIZE, TARGET_SIZE))
            pil_image_resize.save(filename)
            sample_count += 1
        except:
            print('Fail to download the image.')
            error_li.append(str(df.url.iloc[i]))
            error_download += 1
            pass
    return error_download, error_li, sample_count

df = pd.read_csv('train.csv')
create_file("""train""")
error_download, error_li, sample_count = image_download(df)
print(error_download, error_li, sample_count)

