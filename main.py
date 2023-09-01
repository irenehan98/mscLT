# This is a sample Python script.
import logging
import sys
from os import path

from utils.dataset import *


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def load(dataset_type, load_all=False):
    home_dir = '/media/pw/TeamGroup/dataset/ds2_' + dataset_type
    dataset = DeepScores(home_dir, dataset_type)
    dataset.load_annotations(annotation_set_filter='deepscores', load_all=load_all)
    return home_dir, dataset


def crop_complete(dataset, home_dir):
    dataset.crop_all_to_instances(out_dir=path.join(home_dir, 'cropped'), resize=(150, 150), bg_opacity=25)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO logging
    # root_dir, ds = load('complete')
    # crop_complete(ds, root_dir)
    root_dir, ds = load('dense', True)
    ds.crop_all_to_instances(out_dir=root_dir + '/cropped', resize=(150, 150), bg_opacity=25)
    # ds.generate_annotations()
    # print(ds.sort_cat_instances())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
