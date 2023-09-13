# This is a sample Python script.
from utils import *


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def prepare_deepscores(dataset_type):
    root = '/media/pw/TeamGroup/dataset'
    # root = '/home/pw/Repositories/dataset'
    out = '/media/pw/TOSHIBA/ds2_complete'
    ds = DeepScores(root, dataset_type, out_dir=out)
    ds.load_annotations(annotation_set_filter='deepscores', load_all=False)

    ds.crop_all_to_instances('cropped', resize=(150, 150), bg_opacity=25,
                             sav='newsav.json')
    ds.generate_annotations('cropped')
    generate_ride_annotations_from_bbn(ds.home_dir)
    return ds


def generate_ride_annotations_from_bbn(home_dir):
    convert_bbn_to_ride(home_dir, 'test.json', 'test.txt')
    convert_bbn_to_ride(home_dir, 'val.json', 'val.txt')
    convert_bbn_to_ride(home_dir, 'train.json', 'train.txt')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO logging
    prepare_deepscores('complete')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
