import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((150, 150, 3), dtype=np.uint8)

img[:50, :] = [255, 0, 0]
img[50:100, :] = [0, 255, 0]
img[100:, :] = [0, 0, 255]
img[55:95, 55:95] = [255, 255, 255]

cropped_region = img[40:110, 40:110]

red_intensity = img[:, :, 0]
green_intensity = img[:, :, 1]
blue_intensity = img[:, :, 2]

only_red = img.copy()
only_red[:, :, 1] = 0
only_red[:, :, 2] = 0

only_green = img.copy()
only_green[:, :, 0] = 0
only_green[:, :, 2] = 0

only_blue = img.copy()
only_blue[:, :, 0] = 0
only_blue[:, :, 1] = 0

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(img)
axes[0, 0].set_title("1. Original Image")

axes[0, 1].imshow(cropped_region)
axes[0, 1].set_title("2. Cropped Slice [40:110, 40:110]")

axes[0, 2].imshow(red_intensity, cmap='gray')
axes[0, 2].set_title("3. Red 2D Intensity Sheet")

axes[1, 0].imshow(only_red)
axes[1, 0].set_title("4. Isolated Red Channel")

axes[1, 1].imshow(only_green)
axes[1, 1].set_title("5. Isolated Green Channel")

axes[1, 2].imshow(only_blue)
axes[1, 2].set_title("6. Isolated Blue Channel")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig("slicing_and_channels_output.png", dpi=150)
print("Complete! Output 'slicing_and_channels_output.png' me save ho gaya hai.")