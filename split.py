import random
from pathlib import Path
import os
import shutil

random.seed(1234)

u = list(range(900))

train = random.sample(u, 600)
test = list(set(u)-set(train))

gtsdb = Path('GTSDB')
train_dir = gtsdb/'train'
train_dir.mkdir(parents=True, exist_ok=True)
test_dir = gtsdb/'test'
test_dir.mkdir(parents=True, exist_ok=True)

for i in train:
    shutil.copy(gtsdb/f'{i:05}.jpg', train_dir)
    shutil.copy(gtsdb/f'{i:05}.txt', train_dir)

for i in test:
    shutil.copy(gtsdb/f'{i:05}.jpg', test_dir)
    shutil.copy(gtsdb/f'{i:05}.txt', test_dir)