import pandas as pd
from pathlib import Path
from PIL import Image

df = pd.read_csv("gt.txt", sep=';', names=("image", "leftCol", "topRow", "rightCol", "bottomRow", "ClassID"))
df['x_center'] = (df['leftCol'] + df['rightCol']) / 2
df['width'] = df['rightCol'] - df['leftCol']
df['y_center'] = (df['topRow'] + df['bottomRow']) / 2
df['height'] = df['bottomRow'] - df['topRow']

image_widths = {}
image_heights = {}

for i in range(900):
    (Path('GTSDB')/f'{i:05}.txt').write_text('')
    im = Image.open(Path('GTSDB')/f'{i:05}.jpg')
    width, height = im.size
    image_widths[f'{i:05}.ppm'] = width
    image_heights[f'{i:05}.ppm'] = height

df['image_width'] = df['image'].map(image_widths)
df['image_height'] = df['image'].map(image_heights)

df['x_center'] /= df['image_width']
df['width'] /= df['image_width']
df['y_center'] /= df['image_height']
df['height'] /= df['image_height']

for name, group in df.groupby("image"):
    group.to_csv((Path('GTSDB')/name).with_suffix('.txt'), sep=' ', columns=['ClassID', 'x_center', 'y_center', 'width', 'height'], header=False, index=False)
