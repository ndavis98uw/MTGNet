import numpy as np
from itertools import compress
import cv2
import h5py
from glob import glob
import os
import pickle
COLOR_TO_INT = {'W': 0, 'U': 1, 'B': 2, 'R': 3, 'G': 4, 'C': 5}
IMG_DIR = 'card_images'
WIDTH = 620
HEIGHT = 450
CHANNELS = 3


def collapse_types(types_file, out):
    types = np.load(types_file, allow_pickle=True)
    col_types = []
    for list in types:
        col_types.append(list[0])
    np.save(out, col_types)


def generate_labels(label_file):
    labels = np.load(label_file)
    label_dict = {}
    label_num = 0
    for label in labels:
        if label not in label_dict.keys():
            label_dict[label] = label_num
            label_num += 1
    return label_dict



def types_tally(col_types_file):
    col_types = np.load(col_types_file)
    types_count = {}
    for single_type in col_types:
        if single_type not in types_count.keys():
            types_count[single_type] = 1
        else:
            types_count[single_type] += 1
    return types_count

def cmc_to_int(cmc_file, out):
    cmc = np.load(cmc_file)
    int_cmc = []
    for f in cmc:
        int_cmc.append(int(f))
    np.save(out, int_cmc)


def color_to_int(single_color_file, out):
    single_color = np.load(single_color_file, allow_pickle=True)
    int_color = []
    for c in single_color:
        if len(c) == 1:
            int_color.append(COLOR_TO_INT[c[0]])
        else:
            int_color.append(COLOR_TO_INT['C'])
    np.save(out, int_color)


def filter_data(colors_file, ids_file, cmc_file, types_file, names_file, boolean_list, colors_out=None, ids_out=None,
                cmc_out=None, types_out=None, names_out=None):
    colors = np.load(colors_file, allow_pickle=True)
    ids = np.load(ids_file)
    cmc = np.load(cmc_file)
    types = np.load(types_file, allow_pickle=True)
    names = np.load(names_file)
    colors = list(compress(colors, boolean_list))
    ids = list(compress(ids, boolean_list))
    cmc = list(compress(cmc, boolean_list))
    types = list(compress(types, boolean_list))
    names = list(compress(names, boolean_list))
    if colors_out:
        np.save(colors_out, colors)
    else:
        np.save(colors_file, colors)
    if ids_out:
        np.save(ids_out, ids)
    else:
        np.save(ids_file, ids)
    if cmc_out:
        np.save(cmc_out, cmc)
    else:
        np.save(cmc_file, cmc)
    if types_out:
        np.save(types_out, types)
    else:
        np.save(types_file, types)
    if names_out:
        np.save(names_out, names)
    else:
        np.save(names_file, names)


def filter_process(colors_file, ids_file, cmc_file, types_file, names_file, boolean_list,  colors_out=None, ids_out=None,
                cmc_out=None, types_out=None, names_out=None):
    filter_data(colors_file, ids_file, cmc_file, types_file, names_file, boolean_list,  colors_out=colors_out, ids_out=ids_out,
                cmc_out=cmc_out, types_out=types_out, names_out=names_out)
    if colors_out:
        color_to_int(colors_out, colors_out)
    else:
        color_to_int(colors_file, colors_out)
    if cmc_out:
        cmc_to_int(cmc_out, cmc_out)
    else:
        cmc_to_int(cmc_file, cmc_out)
    if types_out:
        collapse_types(types_out, types_out)
    else:
        collapse_types(types_file, types_file)


def to_h5(data_dir, label_type):
    labels = np.load(os.path.join(data_dir, label_type + '.npy'))
    ids = np.load(os.path.join(data_dir, 'ids.npy'))
    names = np.load(os.path.join(data_dir, 'names.npy'))
    with h5py.File(data_dir + '/data.h5', 'w') as hf:
        for (i, id) in enumerate(ids):
            image = cv2.imread(os.path.join(IMG_DIR, id+'.png'))
            image = cv2.resize(image, (HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
            Xset = hf.create_dataset(
                name='X_' + id,
                data=image,
                shape=(HEIGHT, WIDTH, CHANNELS),
                maxshape=(HEIGHT, WIDTH, CHANNELS),
                compression="gzip",
                compression_opts=9)
            yset = hf.create_dataset(
                name='y_' + id,
                data=labels[i],
                shape=(1, ),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)


def reduce_images(height, width, in_dir, out_dir):
    images = glob(os.path.join(in_dir, "*.png"))
    for img in images:
        image = cv2.imread(img)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        fname = img.split('/')[1]
        cv2.imwrite(out_dir + '/' + fname, image)


# colors = np.load('raw_data/colors.npy', allow_pickle=True)
# filter_process('raw_data/colors.npy', 'raw_data/ids.npy', 'raw_data/cmc.npy', 'raw_data/types.npy', 'raw_data/names.npy', list(map(lambda c: len(c) < 3, colors)),
#                 colors_out='dualcolor/colors.npy', ids_out='dualcolor/ids.npy', cmc_out='dualcolor/cmc.npy', types_out='dualcolor/types.npy', names_out='monocolor/names.npy')
# types_count = types_tally('monocolor/types.npy')
# types = np.load('monocolor/types.npy')
# filter_data('monocolor/colors.npy', 'monocolor/ids.npy', 'monocolor/cmc.npy', 'monocolor/types.npy', 'monocolor/names.npy', list(map(lambda t: types_count[t] >= 10, types)),
#                 colors_out='types_10/colors.npy', ids_out='types_10/ids.npy', cmc_out='types_10/cmc.npy', types_out='types_10/types.npy', names_out='types_10/names.npy')
# filter_data('monocolor/colors.npy', 'monocolor/ids.npy', 'monocolor/cmc.npy', 'monocolor/types.npy', 'monocolor/names.npy', list(map(lambda t: types_count[t] >= 50, types)),
#                 colors_out='types_50/colors.npy', ids_out='types_50/ids.npy', cmc_out='types_50/cmc.npy', types_out='types_50/types.npy', names_out='types_50/names.npy')
# filter_data('monocolor/colors.npy', 'monocolor/ids.npy', 'monocolor/cmc.npy', 'monocolor/types.npy', 'monocolor/names.npy', list(map(lambda t: types_count[t] >= 100, types)),
#                 colors_out='types_100/colors.npy', ids_out='types_100/ids.npy', cmc_out='types_100/cmc.npy', types_out='types_100/types.npy', names_out='types_100/names.npy')
# reduce_images(HEIGHT, WIDTH, 'card_images', 'reduced_images')
dict = generate_labels('monocolor/types_100/types.npy')
with open('monocolor/types_100/types.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

