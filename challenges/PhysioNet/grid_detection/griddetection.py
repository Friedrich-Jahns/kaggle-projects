import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import napari

def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def binary(img):
    binary_img = np.where(img>.5,0,1)
    return binary_img



def connectet_components(img):
    shape = np.array(img).shape
    cc_img = np.zeros(shape).astype(int)
    cc_counter = 1
    equiv_list = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i,j] != 0:
                if cc_img[i-1,j] != 0:
                    cc_img[i,j] = cc_img[i-1,j]
                elif cc_img[i,j-1] != 0:
                    cc_img[i,j] = cc_img[i,j-1]
                else:
                    cc_img[i,j] = cc_counter
                    cc_counter += 1
                if cc_img[i-1,j] != 0 and cc_img[i,j-1] != 0 and cc_img[i-1,j] != cc_img[i,j-1]:
                    equiv_list.append([cc_img[i-1,j],cc_img[i,j-1]])
    equiv_list = np.array(equiv_list).astype(int)
    equiv_list.sort(axis=0)
    # for i in range(equiv_list.max())
    # for i in range(shape[0]):
    #     for j in range(shape[1]):

    
    
    return cc_img,equiv_list,cc_counter



img_path = Path.cwd().parent/'data_sample'/'1006427285'/'1006427285-0001.png'

img = plt.imread(img_path)

img = img[600:800,50:500]

img = grayscale(img)

img = binary(img)

img = connectet_components(img)[0]
print(connectet_components(img)[1])


viewer = napari.Viewer()
viewer.add_image(img)

napari.run()


# plt.imshow(img,cmap='gray')
# plt.show()
