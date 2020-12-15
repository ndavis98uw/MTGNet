import numpy as np
from itertools import compress
import cv2
import h5py
from glob import glob
import os
import pickle
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


def color_to_int(color_file, out, label_out):
    color = np.load(color_file, allow_pickle=True)
    int_color = []
    color_labels = {}
    label = 0
    for c in color:
        c_str = ""
        for el in c:
            c_str = c_str + el
        if c_str not in color_labels:
            color_labels[c_str] = label
            label = label + 1
        int_color.append(color_labels[c_str])
    # np.save(out, int_color)
    print(color_labels)
    if label_out:
        with open(label_out, 'wb') as handle:
            pickle.dump(color_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)



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
                cmc_out=None, types_out=None, names_out=None, label_out=None):
    filter_data(colors_file, ids_file, cmc_file, types_file, names_file, boolean_list,  colors_out=colors_out, ids_out=ids_out,
                cmc_out=cmc_out, types_out=types_out, names_out=names_out)
    if colors_out:
        color_to_int(colors_out, colors_out, label_out)
    else:
        color_to_int(colors_file, colors_out, label_out)
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
        fname = img.split('\\')[1]
        cv2.imwrite(out_dir + '\\' + fname, image)


def test_train_split(label_dir, label_type, test_train_split):
    ids = np.load(label_dir + '/ids.npy')
    labels = np.load(label_dir + '/' + label_type + '.npy')
    indices = np.arange(ids.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:int(len(indices) * test_train_split)]
    np.save(label_dir + '\\' + 'ids_train.npy', ids[train_indices])
    np.save(label_dir + '\\' + label_type + '_train.npy', labels[train_indices])
    test_indices = indices[int(len(indices) * test_train_split):]
    np.save(label_dir + '\\' + 'ids_test.npy', ids[test_indices])
    np.save(label_dir + '\\' + label_type + '_test.npy', labels[test_indices])


colors = np.load('raw_data_new/colors.npy', allow_pickle=True)

filter_process('raw_data_new/colors.npy', 'raw_data_new/ids.npy', 'raw_data_new/cmc.npy', 'raw_data_new/types.npy', 'raw_data_new/names.npy', list(map(lambda c: len(c) < 2, colors)),
                colors_out='monocolor_new/colors.npy', ids_out='monocolor_new/ids.npy', cmc_out='monocolor_new/cmc.npy', types_out='monocolor_new/types.npy', names_out='monocolor_new/names.npy')
# types_count = types_tally('monocolor_new/types.npy')
# types = np.load('monocolor_new/types.npy')
# filter_data('monocolor_new/colors.npy', 'monocolor_new/ids.npy', 'monocolor_new/cmc.npy', 'monocolor_new/types.npy', 'monocolor_new/names.npy', list(map(lambda t: types_count[t] >= 100, types)),
#                 colors_out='monocolor_new/types_100/colors.npy', ids_out='monocolor_new/types_100/ids.npy', cmc_out='monocolor_new/types_100/cmc.npy', types_out='monocolor_new/types_100/types.npy', names_out='monocolor_new/types_100/names.npy')
# reduce_images(315, 434, 'card_images_new', 'extra_reduced_images_new')
#dict = generate_labels('monocolor_new/colors.npy')
print("hello")
# with open('monocolor_new/colors.pickle', 'wb') as handle:
#     pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
filter_process('raw_data_new/colors.npy', 'raw_data_new/ids.npy', 'raw_data_new/cmc.npy', 'raw_data_new/types.npy', 'raw_data_new/names.npy', list(map(lambda c: len(c) < 3, colors)),
                colors_out='dualcolor_new/colors.npy', ids_out='dualcolor_new/ids.npy', cmc_out='dualcolor_new/cmc.npy', types_out='dualcolor_new/types.npy', names_out='dualcolor_new/names.npy')
# filter_data('dualcolor_new/colors.npy', 'dualcolor_new/ids.npy', 'dualcolor_new/cmc.npy', 'dualcolor_new/types.npy', 'dualcolor_new/names.npy', list(map(lambda t: types_count[t] >= 100, types)),
#                 colors_out='dualcolor_new/types_100/colors.npy', ids_out='dualcolor_new/types_100/ids.npy', cmc_out='dualcolor_new/types_100/cmc.npy', types_out='dualcolor_new/types_100/types.npy', names_out='dualcolor_new/types_100/names.npy')
# filter_data('dualcolor_new/colors.npy', 'dualcolor_new/ids.npy', 'dualcolor_new/cmc.npy', 'dualcolor_new/types.npy', 'dualcolor_new/names.npy', list(map(lambda t: types_count[t] >= 400, types)),
#                 colors_out='dualcolor_new/types_400/colors.npy', ids_out='dualcolor_new/types_400/ids.npy', cmc_out='dualcolor_new/types_400/cmc.npy', types_out='dualcolor_new/types_400/types.npy', names_out='dualcolor_new/types_400/names.npy')
# test_train_split('monocolor_new', 'colors', .8)
# test_train_split('monocolor_new/types_100', 'types', .8)
# test_train_split('dualcolor_new', 'colors', .8)
# test_train_split('dualcolor_new/types_100', 'types', .8)

