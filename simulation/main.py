import os
import numpy as np
import matplotlib.pyplot as plt
import SimulateTool as Simtool
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# How many images to simulate
images = 2
columnlen = 2  # images >= columnlen (must)

# If you want to print values or plot images
Plot = True
load_image = False
Print = False

if load_image:
    create_images = images - 1
else:
    create_images = images

# ST objekt
st = Simtool.SimulateTool(numImages=create_images, numParticlesRange=[35, 40], limit=15, imageSize=[1944, 1458],
                          addAugment=False)

dt_image = st.getImages()
plot_images = dt_image

if load_image:
    path = r'C:\Users\darak\PycharmProjects\Kandidatarbete\Github\Kandidatarbete\ArashLocal\exp_image0.npy'
    loaded_image = np.load(path)
    loaded_image = 1 + loaded_image[:, :, 0] + 1j * loaded_image[:, :, 1]
    plot_images.append(loaded_image)

if Print:
    illummatrix = {}
    for q in range(images):
        elem = 'Image' + str(1 + q)
        illummatrix[elem] = st.getIllumination(dt_image[q][0]) * 1e5
    print(illummatrix)

rows = math.ceil(images / columnlen)
if Plot:
    if images == 1:
        plt.imshow(np.abs(plot_images[0]), cmap='gray')
    elif rows == 1:
        fig, ax = plt.subplots(1, columnlen)
        k = 0
        for j in range(columnlen):
            if k < images:
                ax[j].imshow(np.abs(plot_images[k]), cmap='gray')
            else:
                ax[j].axis('off')
            k += 1
    else:
        fig, ax = plt.subplots(rows, columnlen)
        k = 0
        for i in range(rows):
            for j in range(columnlen):
                if k < images:
                    ax[i, j].imshow(np.abs(plot_images[k]), cmap='gray')
                else:
                    ax[i, j].axis('off')
                k += 1
    plt.show()
