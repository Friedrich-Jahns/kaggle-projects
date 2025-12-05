import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import napari
import cc3d

def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def binary(img):
    binary_img = np.where(img>.8,0,1)
    return binary_img

def selctive_binary(img,val):
    binary_img = np.where(img==val,0,1)
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
    equiv_list_sorted = []
    for i,val in enumerate(equiv_list):
        if i == 0 or not np.array_equal(val, equiv_list[i-1]):
            equiv_list_sorted.append(val)

        # if i != 0 and all(equiv_list[i-1] == equiv_list[i]):
        #         # print(f'dubble{equiv_list[i-1]}={equiv_list[i]}')
        #         continue
        # else:
        #     print('here')
        #     equiv_list_sorted.append(val)
    for i in range(1):
        for eqiv in equiv_list_sorted:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if cc_img[i,j] == eqiv[0]:
                        cc_img[i,j] = eqiv[1]




    return cc_img,np.array(equiv_list_sorted),cc_counter



# img_path = Path.cwd()/'data_sample'/'1006427285'/'1006427285-0001.png'

img_path = Path('/home/friedrich/Dokumente/Software/kaggle-projects/challenges/PhysioNet/data_sample') / '1006427285' /'1006427285-0009.png'



img = plt.imread(img_path)

img = img[600:1100,50:900]

img = grayscale(img)

img = binary(img)

img = connectet_components(img)[0]
# labels = cc3d.connected_components(img)
# print(connectet_components(img)[1])


viewer = napari.Viewer()
viewer.add_image(img)
nonzero_pixels = img[img > 0]
values, counts = np.unique(nonzero_pixels, return_counts=True)
most_frequent_value = values[np.argmax(counts)]
print(most_frequent_value)
viewer.add_image(selctive_binary(img,most_frequent_value))


napari.run()


# plt.imshow(img,cmap='gray')
# plt.show()


labels, N = cc3d.connected_components(img, return_N=True)

# Größte Komponente (außer Hintergrund, Label 0)
counts = np.bincount(labels.flat)  # zählt Pixel pro Label
counts[0] = 0  # Hintergrund ignorieren
largest_label = np.argmax(counts)

# Bild der größten Komponente erstellen
largest_component = (labels == largest_label).astype(np.uint8)

# Anzeigen
plt.imshow(largest_component, cmap='gray')
plt.title(f'Größte Komponente: Label {largest_label}')
plt.show()