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


cwd = Path(os.path.dirname(os.path.abspath(__file__)))


#dat_path = Path.cwd().parent.parent/'data_sample'
#print(dat_path)

#save_path = Path.cwd().parent/ 'train_data' / 'train'
#print(save_path)

dat_path = cwd.parent.parent/'data_sample'
print(dat_path)

save_path = cwd.parent/ 'train_data' / 'train'
print(save_path)


for i in os.listdir(dat_path):
    filepath = dat_path / i / f'{i}-0001.png'
    data  = mask_gererator(filepath)
    curves_and_text = data.segment_curves_and_text()
    if False:
        print(i)
        plt.figure(figsize=(15,15))
        plt.subplot(331)
        plt.imshow(data.image,cmap='gray')
        plt.title('Original')
        plt.subplot(332)
        plt.imshow(data.image_r,cmap='gray')
        plt.title('Red-Channel')
        plt.subplot(333)
        plt.imshow(data.grid,cmap='gray')
        plt.title('Grid = Red-Channel - Blue_Channel')
        plt.subplot(336)
        plt.imshow(data.grid_mask(),cmap='gray')
        plt.title('Grid_binary')
        plt.subplot(334)
        plt.imshow(curves_and_text[1],cmap='gray')
        plt.title('thresholded_text')
        plt.subplot(335)
        plt.imshow(curves_and_text[2],cmap='gray')
        plt.title('thresholded_curve')
        plt.subplot(338)
        plt.imshow(curves_and_text[3],cmap='gray')
        plt.title('thresholded_curve+CC_filter')
        plt.show()
    if True:
        for sub in ['curve_mask', 'text_mask', 'img', 'grid_mask']:
            (save_path / sub).mkdir(parents=True, exist_ok=True)


        mpimg.imsave(save_path / 'curve_mask' / f'{i}.png', curves_and_text[3], cmap='gray')
        mpimg.imsave(save_path / 'text_mask' / f'{i}.png', curves_and_text[1], cmap='gray')
        mpimg.imsave(save_path / 'img' / f'{i}.png', data.image, cmap='gray')
        mpimg.imsave(save_path / 'grid_mask' / f'{i}.png', data.grid_mask(), cmap='gray')


