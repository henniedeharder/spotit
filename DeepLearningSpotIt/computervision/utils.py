import os
from os import listdir, makedirs
from os.path import isfile, isdir, join

def list_files_in_dir(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]

def create_prediction_dirs(nr):
    dirname = f'test/predict{nr}'
    if not isdir(dirname):
        makedirs(dirname)
        makedirs(f'{dirname}/predict')
    return dirname

def get_labels():
    names = sorted(['hammer', 'lightning', 'fire', 'ghost', 'key', 'cat', 'icecube', 'snowflake', 
          'turtle', 'milkbottle', 'eye', 'clown', 'cactus', 'gkey', 'cobweb', 'lightbulb', 
          'carrot', 'hand', 'bird', 'stopsign', 'igloo', 'lips', 'flower', 'exclamationmark',
          'car', 'lock', 'anchor', 'moon', 'man', 'clock', 'tree', 'heart', 'spider', 'stains', 
          'dolphin', 'apple', 'ladybug', 'trex', 'sun', 'cheese', 'questionmark', 'dog', 'horse', 
          'flyingdino', 'zebra', 'yinyang', 'sunglasses', 'skull', 'candle', 'snowman', 'leaf', 
          'drop', 'bomb', 'scissors', 'pencil', 'bullseye', 'clover'])

    labels = dict()

    for i in range(len(names)):
        labels[i] = names[i]

    return labels

def indices_to_labels(predicted_class_indices):
    labels = get_labels()
    return [labels[k] for k in predicted_class_indices]