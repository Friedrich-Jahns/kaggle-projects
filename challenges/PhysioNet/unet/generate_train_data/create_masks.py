from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import matplotlib.image as mpimg

class mask_gererator:
    def __init__(self, image_path):
        self.image = plt.imread(image_path)
        self.image_r = self.image[:,:,0]
        self.image_g = self.image[:,:,1]
        self.image_b = self.image[:,:,2]
        self.image_a = self.image[:,:,3]

        curves_and_text = self.image_r

        self.grid = self.image_b-curves_and_text

    def grid_mask(self):
        grid_binary = np.where(self.grid<-.5,1,0)
        return grid_binary

    def cc_filter(self,img,min_size):
        img = img.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        filtered_mask = np.zeros_like(img)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                filtered_mask[labels == i] = 1

        return filtered_mask


    def segment_curves_and_text(self):
        print(np.median(self.image_r))
        binary_c_t = np.where(self.image_r<.99,1,0)
        binary_t = np.where(self.image_r<.0001,1,0)
        binary_c = binary_c_t-binary_t
        filtered_c = self.cc_filter(binary_c,200)
        return binary_c_t, binary_t, binary_c, filtered_c




dat_path = Path.cwd().parent.parent/'data_sample'
print(dat_path)

save_path = Path.cwd().parent/ 'train_data' / 'train'
print(save_path)

for i in os.listdir(dat_path):
    filepath = dat_path / i / f'{i}-0001.png'
    data  = mask_gererator(filepath)
    curves_and_text = data.segment_curves_and_text()
    if False:
        print(i)
        plt.figure(figsize=(15,15))
        plt.subplot(221)
        plt.imshow(data.image,cmap='gray')
        plt.subplot(222)
        plt.imshow(data.image_r,cmap='gray')
        plt.subplot(223)
        plt.imshow(curves_and_text[1],cmap='gray')
        plt.subplot(224)
        plt.imshow(curves_and_text[3],cmap='gray')
        plt.show()
    if True:

        mpimg.imsave(save_path / 'curve_mask' / f'{i}.png', curves_and_text[3], cmap='gray')
        mpimg.imsave(save_path / 'text_mask' / f'{i}.png', curves_and_text[1], cmap='gray')
        mpimg.imsave(save_path / 'img' / f'{i}.png', data.image, cmap='gray')
        mpimg.imsave(save_path / 'grid_mask' / f'{i}.png', data.grid_mask(), cmap='gray')


