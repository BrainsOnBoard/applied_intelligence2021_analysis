# %%
import sys
NAVBENCH_PATH = '../navbench'
sys.path.append(NAVBENCH_PATH)

import cv2
import matplotlib.pyplot as plt
import navbench as nb
from navbench import improc as ip
import os

plt.rcParams.update({'font.size': 18})

preprocess = (ip.histeq, ip.to_float)
image = nb.read_images(os.path.join(NAVBENCH_PATH, 'datasets/panorama_grand_canyon.jpg'), preprocess)
diffs = nb.ridf(image, [image])

fig, ax = plt.subplots(figsize=(7, 2))


nb.plot_ridf(diffs, ax=ax)
ax.set_ylabel('Image difference');
ax.set_xlabel('Orientation (°)')
fig.savefig('figures/ridf.svg', bbox_inches='tight')
