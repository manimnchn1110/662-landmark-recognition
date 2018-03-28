import numpy as np
from PIL import Image



def convert_rgb(img_name='b399f09dee9c3c67.jpeg'):
    img=Image.open(img_name)
    image_rgb = img.convert("RGB")    #convert image into RGB pixel
    #image_bw = img.convert("L")# ---- B/L pixel
    # convert image to a matrix with values from 0 to 255 (uint8)
    rgb_arr = np.asarray(image_gr)
    print("After conversion to numerical representation: \n\n %r" % rgb_arr)
    %matplotlib inline

convert_rgb(img_name)
