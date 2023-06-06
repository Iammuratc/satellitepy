import cv2
import matplotlib.pyplot as plt
import json


image_path = '/home/murat/Projects/satellitepy/data/DOTA/temp/images/P0004_x_0_y_502.png'
label_path = '/home/murat/Projects/satellitepy/data/DOTA/temp/labels/P0004_x_0_y_502.json'


img = cv2.imread(image_path)

with open(label_path,'r') as f:
    labels = json.load(f)

fig, ax = plt.subplots(2)

ax[0].imshow(img)
ax[1].imshow(img)

for mask_indices in labels['masks']:
    ax[1].plot(mask_indices[0],mask_indices[1])
plt.show()