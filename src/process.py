from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

splattern_types = ['castoff', 'drops', 'projected', 'spatters']

jpg_files = [str(p) for splattern in splattern_types for p in Path(f'./images/raw/{splattern}/').glob('*.jpg')]

for filename in jpg_files:
    print(f'Processing file {filename}')
    im = Image.open(filename)
    brightness = ImageEnhance.Brightness(im)
    im = brightness.enhance(2)
    contrast = ImageEnhance.Contrast(im)
    im = contrast.enhance(1.5)
    im
    im = im.convert('RGBA')
    data = np.array(im)
    rgb = data[:, :, :3]
    color = [50, 0, 0]
    black = [0, 0, 0, 255]
    white = [255, 255, 255, 255]
    mask = np.all((rgb - color) > 50, axis=-1)
    data[mask] = white
    new_im = Image.fromarray(data)
    grayscale = new_im.convert("L")
    grayscale.save(filename.replace('/raw/', '/processed/').replace('.jpg', '') + '-processed.jpg')
