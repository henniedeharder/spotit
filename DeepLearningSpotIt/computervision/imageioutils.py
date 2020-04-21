from .utils import list_files_in_dir
import imageio
from os.path import join

def create_gif(directory, name):
    files = list_files_in_dir(directory)
    images = []
    for filename in files:
        images.append(imageio.imread(join(directory, filename)))
    imageio.mimsave(f'gifs/{name}.gif', images, duration=0.5)