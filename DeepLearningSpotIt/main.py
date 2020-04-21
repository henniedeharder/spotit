from computervision.opencvutils import Image
from computervision.utils import *
from computervision.kerasutils import *
from computervision.imageioutils import *

# remove
import time
from os.path import join
import shutil
import cv2
import itertools

start = time.perf_counter()

gif = False
remove = True

# load CNN model
model = load_cnn_model()

# get images
directory = 'predict'
files = list_files_in_dir(directory)

# create list with image objects to handle
images = [Image(path_to_img=join(directory, files[i])) for i in range(len(files))]

nr = 0
for img in images:
    # process files for predicting by adding contrast, resize, blur
    img.processed = img.contrast_resized_blurred()
    # create prediction directories
    img.preddir = create_prediction_dirs(nr)    
    # grab biggest contour: this is the card
    cnt = Image.grab_contours_by_area(img.processed)[0]
    img.card = Image.keep_contour_with_white_background(img.processed, cnt)
    # grab 10 largest contours, at least you have the eight icons on the card
    img.cnts = Image.grab_contours_by_area(img.card, reverse=True, threshold=190, area=900)[:10]
    
    i = 0
    for c in img.cnts:
        x, y, w, h = Image.get_rect_coordinates_around_contour(c)
        img.cntsx.append(x)
        img.cntsy.append(y)
        img.drawncontour = Image.draw_contour(img.processed.copy(), c)
        img.cnts_images.append(img.drawncontour)
        img.icon = Image.keep_contour_with_white_background(img.card, c)
        x, y, w, h = Image.bounding_square_around_contour(c)
        img.roi = Image.take_out_roi(img.icon, x, y, w, h)
        img.roi = Image.resize_image(img.roi, (400,400))
        img.save_image(f'test/predict{nr}/predict', img.roi, addition=f'_{i}') 
        i += 1
    
    # predict
    img.predicted_class_indices, img.predicted_probabilities = get_predictions(img.preddir, model)
    img.predicted_labels = indices_to_labels(img.predicted_class_indices)
    for i in range(len(img.predicted_labels)):
        img.predictions[img.predicted_labels[i]] = i

    for i in range(len(img.cnts_images)):
        probability = '%.2f' %img.predicted_probabilities[i]
        text = f'{img.predicted_labels[i].capitalize()}: {probability}'
        img.cnts_images[i] = Image.add_text(img.cnts_images[i], text, x=img.cntsx[i]-20, y=img.cntsy[i]-10)
        img.save_image(directory=f'test/predict{nr}', image=img.cnts_images[i], addition=f'_{i}')

    if gif:    
        create_gif(img.preddir, f'{img.wo_extension}')

    nr += 1

for combo in itertools.combinations(images, 2):
    all_predictions = [img.predicted_labels for img in combo]

    common = list(set(all_predictions[0]).intersection(all_predictions[1]))
    probs_means = []
    mean = 0
    common_icon = []
    if len(common) == 1:
        common_icon.append(common[0])
    if len(common) > 1:
        for icon in common:     
            mean = np.mean([img.predicted_probabilities[img.predictions[icon]] for img in combo])
            probs_means.append(mean)
        idx = probs_means.index(max(probs_means))
        common_icon.append(common[idx])

    add_images = []
    found_name = ''

    if len(common_icon) == 1:
        text = f'The common icon is: {common_icon[0].capitalize()}!'
        for img in combo:
            winning = img.cnts_images[img.predictions[common_icon[0]]]
            add_images.append(winning)
            found_name += img.wo_extension
        found = Image.add_2_images(add_images[0], add_images[1])
        found = Image.add_text(found, text, x=600, y=50, thickness=4)
        Image.save_image_(directory='predicted', image=found, name=f'{common_icon[0]}{found_name}')
    else:
        text = f"Didn't find icons :("
        for img in combo:
            add_images.append(img.processed)
            found_name += img.wo_extension
        found = Image.add_2_images(add_images[0], add_images[1])
        found = Image.add_text(found, text, x=600, y=50, thickness=4)
        Image.save_image_(directory='predicted', image=found, name=f'01_{found_name}')

if remove:
    for img in images:
        shutil.rmtree(img.preddir)

end = time.perf_counter()
totaltime = end - start

print(f'To find all the Spot it! combinations took the computer {totaltime} seconds!')