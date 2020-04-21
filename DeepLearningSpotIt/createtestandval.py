import os, shutil
from os import listdir
from os.path import isfile, join
import random
import shutil

original_dir = 'icons/nobluricons'
train = join(original_dir, 'train')
test = join(original_dir, 'test')
val = join(original_dir, 'validation')

dirs = [f for f in listdir(train)]

for dr in dirs:
    # randomly select four files for validation set
    files = [f for f in listdir(join(train, dr)) if isfile(join(train, dr, f))]
    filenames = random.sample(files, 4)
    for fn in filenames:
        if not os.path.exists(join(val, dr)):
            os.makedirs(join(val, dr))
        shutil.move(join(train, dr, fn), join(val, dr, fn))
    # same for test set
    files = [f for f in listdir(join(train, dr)) if isfile(join(train, dr, f))]
    filenames = random.sample(files, 4)
    for fn in filenames:
        if not os.path.exists(join(test, dr)):
            os.makedirs(join(test, dr))
        shutil.move(join(train, dr, fn), join(test, dr, fn))
    