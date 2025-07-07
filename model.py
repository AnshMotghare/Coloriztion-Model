import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#PrePared Model
proto = './models/colorization_deploy_v2.prototxt'
weights = './models/colorization_release_v2.caffemodel'


# load cluster centers
pts_in_hull = np.load('./models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

# load model
net = cv2.dnn.readNetFromCaffe(proto, weights)
# net.getLayerNames()

# populate cluster centers as 1x1 convolution kernel
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
# scale layer doesn't look work in OpenCV dnn module, we need to fill 2.606 to conv8_313_rh layer manually
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

# Pre-Processing
img_path = 'img/sample_3_input.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_input = img.copy()

# convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

img_rgb = img.copy()

# normalize input
img_rgb = (img_rgb / 255.).astype(np.float32)

# convert RGB to LAB
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
# only L channel to be used
img_l = img_lab[:, :, 0]



# Resize input to network size and mean-center
img_resized = cv2.resize(img_l, (224, 224))
img_resized = img_resized.astype(np.float32)
img_resized -= 50  # mean-centering

# Prepare network input
net.setInput(cv2.dnn.blobFromImage(img_resized))

# Forward pass
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize ab channels to original image size
ab_dec_us = cv2.resize(ab_dec, (img.shape[1], img.shape[0]))

# Combine original L with predicted ab
l_channel = img_lab[:, :, 0]
lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_dec_us), axis=2)

# Convert back to BGR color space
bgr_output = cv2.cvtColor(lab_output.astype(np.float32), cv2.COLOR_Lab2BGR)
bgr_output = np.clip(bgr_output, 0, 1)  # keep pixel values in range
bgr_output = (255 * bgr_output).astype(np.uint8)

# Show original grayscale and colorized rest
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Input (Grayscale)")
plt.imshow(cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Colorized Output")
plt.imshow(cv2.cvtColor(bgr_output, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()


#plt.subplot(1, 2, 2)
#plt.title("Colorized Output")
#plt.imshow(cv2.cvtColor(bgr_output, cv2.COLOR_BGR2RGB))
#plt.axis('off')

#plt.tight_layout()
#plt.show()


