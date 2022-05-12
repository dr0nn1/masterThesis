#Used for making the images used in the theory chapter "Digital image 3.2.3" figure 3.4 and 3.5 
import cv2
from PIL import Image
import numpy as np

def rgb():
    w, h = 600, 400
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:100, 0:600] = [0, 0, 0]
    data[100:200, 0:600] = [255, 0, 0] 
    data[200:300, 0:600] = [0, 255, 0] 
    data[300:400, 0:600] = [0, 0, 255] 
    #print(data)
    img = Image.fromarray(data, 'RGB')
    img.save('rgb.png')
    img.show()

def gray():
    w, h = 600, 400
    data = np.zeros((h, w), dtype=np.uint8)

    data[0:100, 0:600] = [0] 
    data[100:200, 0:600] = [100] 
    data[200:300, 0:600] = [200]
    data[300:400, 0:600] = [255] 
    data[0:1, 0:600] = [0]
    data[399:400, 0:600] = [0]
    data[0:400, 0:1] = [0]
    data[0:400, 599:600] = [0]
    #print(data)
    img = Image.fromarray(data)
    img.save('gray.png')
    img.show()

if __name__ == "__main__":
    rgb()
    gray()