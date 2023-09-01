"""
partially based on obb_anns (https://github.com/yvan674/obb_anns)
"""
import itertools
import json
import math
import os
import random
from concurrent import futures
from datetime import datetime
from typing import List
from os import path
from tqdm import tqdm
from time import time

from PIL import Image, ImageColor, ImageDraw, ImageFont
import colorcet
import numpy
import pandas


def new_bbn_ann_inst(cat_id, current_id, height, img_fp, width):
    return {
        "image_id": current_id,
        "im_height": height,
        "im_width": width,
        "category_id": cat_id,
        "fpath": img_fp,
    }


def calc_resulting_dim(img_size, bbox, resize):
    if resize is not None:
        return resize

    # TODO unfinished draft
    x0, y0, x1, y1 = bbox
    width, height = int(x1 - x0), int(y1 - y0)
    max_width, max_height = img_size

    return width, height


def get_random_inst_idxs(rand_max, idx_amt):
    idx_set = set()
    while len(idx_set) < idx_amt:
        idx_set.add(random.randint(0, rand_max - 1))
    return idx_set


def generate_train_annotations(cat_instances, excl_sets, cat_info, new_mapping, home_dir, filename, current_id, style):
    print(f"generating {filename}..")

    img_dir = path.join(home_dir, 'cropped')
    annotations = []

    cat_cnt = 0
    for cat, cnt in cat_instances:
        if cnt <= 0:
            continue
        cat_cnt += 1

        name = cat_info[cat]['name']
        cat_id = cat if cat not in new_mapping else new_mapping[cat]
        for idx in range(cnt):
            if idx in excl_sets[cat]:
                continue
            # WARNING: quick implementation only, use your own params!
            width, height = (150, 150)
            # WARNING: len depends on dataset_type
            img_fp = path.join(img_dir, f'{name}-{idx:06d}') + '.png'
            if style == 'bbn':
                annotations.append(new_bbn_ann_inst(cat_id, current_id, height, img_fp, width))
            current_id += 1
    with open(path.join(home_dir, filename), 'w') as fp:
        json.dump({'annotations': annotations, 'num_classes': cat_cnt, "remapped_cats": new_mapping}, fp)
    return


def generate_annotation(idx_sets, cat_info, new_mapping, home_dir, filename, current_id, style):
    print(f"generating {filename}..")
    img_dir = path.join(home_dir, 'cropped')
    annotations = []
    for cat, idx_set in idx_sets:
        name = cat_info[cat]['name']
        # TODO handle new_mapping value None
        cat_id = cat if cat not in new_mapping else new_mapping[cat]
        for idx in idx_set:
            # WARNING: quick implementation only, use your own params!
            width, height = (150, 150)
            # WARNING: len depends on dataset_type
            img_fp = path.join(img_dir, f'{name}-{idx:06d}') + '.png'
            if style == 'bbn':
                annotations.append({
                    "image_id": current_id,
                    "im_height": height,
                    "im_width": width,
                    "category_id": cat_id,
                    "fpath": img_fp,
                })
            current_id += 1
    with open(path.join(home_dir, filename), 'w') as fp:
        json.dump({'annotations': annotations, 'num_classes': len(idx_sets), "remapped_cats": new_mapping}, fp)
    return current_id


def generate_test_val_set(cat_instances, test_coef, val_coef):
    test_sets = dict()
    val_sets = dict()

    for k, cnt in cat_instances:
        if cnt <= 0:
            continue

        test_cnt = max(1, math.floor(cnt * test_coef))
        val_cnt = max(1, math.floor(cnt * val_coef))

        test_sets[k] = get_random_inst_idxs(cnt, test_cnt)
        val_sets[k] = get_random_inst_idxs(cnt, val_cnt)
    return test_sets, val_sets


# TODO get max decimal len
def generate_all_annotations(cat_instances, cat_info, home_dir, style='bbn', new_mapping=None):
    test_set, val_set = generate_test_val_set(cat_instances, 0.2, 0.2)

    current_id = 1
    current_id = generate_annotation(test_set.items(), cat_info, new_mapping, home_dir, 'test.json', current_id, style)
    current_id = generate_annotation(val_set.items(), cat_info, new_mapping, home_dir, 'val.json', current_id, style)

    combined_set = test_set
    for k, v in val_set.items():
        combined_set[k].update(v)
    generate_train_annotations(cat_instances, combined_set, cat_info, new_mapping, home_dir, 'train.json', current_id, style)
    return


# TODO rename to crop_with_bg
def crop_object(img, bbox, min_dim, bg_opacity, resize):
    x0, y0, x1, y1 = bbox
    width, height = int(x1 - x0), int(y1 - y0)
    max_width, max_height = img.size

    bg_wdiff, bg_hdiff = max(0, min_dim - width), max(0, min_dim - height)
    bg_x0, bg_x1 = max(0, x0 - (bg_wdiff / 2)), min(max_width, x1 + (bg_wdiff / 2))
    bg_y0, bg_y1 = max(0, y0 - (bg_hdiff / 2)), min(max_height, y1 + (bg_hdiff / 2))

    cropped = img.crop((x0, y0, x1, y1))
    bg_cropped = img.crop((bg_x0, bg_y0, bg_x1, bg_y1))
    bg_cropped.putalpha(bg_opacity)

    result = Image.new(cropped.mode, bg_cropped.size, (255, 255, 255))
    result.paste(bg_cropped, (0, 0), bg_cropped)
    result.paste(cropped, (int(x0 - bg_x0), int(y0 - bg_y0)))

    if resize is not None:
        result = result.resize(resize)

    return result


class DeepScores:
    def __init__(self, root_dir, dataset_type):
        print("initializing...")
        start_time = time()

        self.dataset_info = None
        self.annotation_sets = None
        self.chosen_ann_set = None  # type: None or List[str]
        self.cat_instance_count = None
        self.ann_infos = None  # type pd dataFrame
        self.cat_infos = None
        self.img_infos = None  # img_info has 'id' 'filename' 'width' 'height' 'ann_ids'
        self.img_idx_lookup = dict()  # lookup table used to figure out the index in img_info of image based on their img_id
        # self.ann_ids = []

        self.root = root_dir
        self.dataset_type = dataset_type
        # TODO: assert dataset exists
        # TODO: find out if deepscores train and test data overlaps
        self.validate_type(dataset_type)

        self.train_ann_files = []
        self.test_ann_files = []
        if dataset_type == 'dense':
            self.train_ann_files.append(path.join(root_dir, "deepscores_train.json"))  # default ann_file
            self.test_ann_files.append(path.join(root_dir, "deepscores_test.json"))
        elif dataset_type == 'complete':
            for i in itertools.count():
                test_fp = path.join(root_dir, f"deepscores-complete-{i}_test.json")
                train_fp = path.join(root_dir, f"deepscores-complete-{i}_train.json")

                if path.isfile(train_fp):
                    self.train_ann_files.append(train_fp)
                if path.isfile(test_fp):
                    self.test_ann_files.append(test_fp)

                if not path.isfile(train_fp) and not path.isfile(test_fp):
                    break

        print("initialization done in {:.6f}s".format(time() - start_time))

    def __repr__(self):
        information = "DeepScore Dataset Annotation\n"
        information += f"root file: {self.root}\n"
        information += f"dataset type: {self.dataset_type}\n"
        if self.dataset_info is not None:
            information += f"Num images: {len(self.img_infos)}\n"
            # TODO count all cat inst cnt for ann count
            information += f"Num anns: {len(self.ann_infos) if self.dataset_type == 'dense' else '-'}\n"
            information += f"Num cats: {len(self.cat_infos)}\n"
            information += f"selected cats: {len(self.get_cats())}"
        else:
            information += "Annotations not yet loaded\n"
        return information

    def __len__(self):
        return 0 if self.img_infos is None else len(self.img_infos)

    @staticmethod
    def _xor_args(m, n):
        only_one_arg = ((m is not None and n is None)
                        or (m is None and n is not None))
        assert only_one_arg, 'Only one type of request can be done at a time'

    @staticmethod
    def validate_type(dataset_type):
        assert dataset_type == 'dense' or dataset_type == 'complete', f"unsupported type: {dataset_type}"

    @staticmethod
    def parse_comments(comment):
        parsed_dict = dict()
        for co in comment.split(";"):
            if len(co.split(":")) > 1:
                key, value = co.split(":")
                parsed_dict[key] = value
        return parsed_dict

    @staticmethod
    def moderate_bbox(bbox, img_size):
        x0, y0, x1, y1 = bbox
        img_width, img_height = img_size

        if x0 == x1:
            x0 = max(0, x0 - 1)
            x1 = min(img_width, x1 + 1)
        if y0 == y1:
            y0 = max(0, y0 - 1)
            y1 = min(img_height, y1 + 1)

        return [x0, y0, x1, y1]

    @staticmethod
    def combine_cat_inst_cnt(inst_cnt, more_cnt):
        # WARNING: unknown categories in muscima++ has None as key values
        for k, v in more_cnt.items():
            inst_cnt[int(k)] += v
        return inst_cnt

    @staticmethod
    def load_annotation_file(ann_fp, cat_idx, with_ann_df=True):
        # start_time = time()
        with open(ann_fp, 'r') as ann_file:
            data = json.load(ann_file)

        img_infos = data['images']

        img_idx_lookup = dict()
        for i, img in enumerate(img_infos):
            img_idx_lookup[int(img['id'])] = i

        ann_ids = []
        annotations = {'a_bbox': [],
                       'o_bbox': [],
                       'cat_id': [],
                       'area': [],
                       'img_id': [],
                       'comments': []}

        cat_inst_count = dict()
        for k, v in data['annotations'].items():
            # TODO find better alternative (e.g. see self.cat_info?)
            cat_id = v['cat_id'][cat_idx]
            cat_inst_count.setdefault(cat_id, 0)
            cat_inst_count[cat_id] += 1

            if with_ann_df is True:
                ann_ids.append(int(k))
                annotations['a_bbox'].append(v['a_bbox'])
                annotations['o_bbox'].append(v['o_bbox'])
                annotations['cat_id'].append(v['cat_id'])
                annotations['area'].append(v['area'])
                annotations['img_id'].append(v['img_id'])
                annotations['comments'].append(v['comments'])

        if with_ann_df is True:
            ann_df = pandas.DataFrame(annotations, ann_ids)
        else:
            ann_df = None

        # class_keys = list(cat_inst_count.keys())
        # self.cat_instance_count = {i: cat_inst_count[i] for i in class_keys}

        # print(f"file {ann_fp} loaded in {time() - start_time:.6f}s")
        return img_infos, img_idx_lookup, cat_inst_count, ann_df

    @staticmethod
    def crop_save_obj_img(img, bbox, resize, min_dim, bg_opacity, out_fp):
        result = crop_object(img, bbox, min_dim, bg_opacity, resize)
        result.save(out_fp)
        return result

    def load_annotations(self, annotation_set_filter=None, load_all=True):
        print("loading annotations...")
        start_time = time()

        # -- only need to load once --
        ref_file = self.test_ann_files[0]

        with open(ref_file, 'r') as ann_file:
            data = json.load(ann_file)

        self.dataset_info = data['info']
        self.annotation_sets = data['annotation_sets']

        # Sets annotation sets and makes sure it exists
        if annotation_set_filter is not None:
            assert annotation_set_filter in self.annotation_sets, \
                f"The chosen annotation_set_filter " \
                f"{annotation_set_filter} is not a in the available " \
                f"annotations sets."
            self.chosen_ann_set = annotation_set_filter
            cat_idx = self.annotation_sets.index(annotation_set_filter)
        else:
            self.chosen_ann_set = self.annotation_sets
            cat_idx = 0  # default deepscores

        self.cat_infos = {int(k): v for k, v in data['categories'].items()}

        cat_instance_count = dict()
        for cat in self.get_cats():
            cat_instance_count.setdefault(cat, 0)

        print(f"basic info loaded in {time() - start_time:.6f}s..")
        # -- end of load once --

        self.img_infos = []

        # press F for my 32gb ram it ain't enough for multithreading
        if load_all:
            with_ann_df = True if self.dataset_type == 'dense' else False
            files = self.train_ann_files + self.test_ann_files
            ann_infos = []
            for ann_file in tqdm(files, 'file processed: ', unit='file'):
                img_infos, img_idx_lookup, new_inst_cnt, ann_df = self.load_annotation_file(ann_file, cat_idx,
                                                                                            with_ann_df)
                self.img_idx_lookup.update(img_idx_lookup)
                cat_instance_count = self.combine_cat_inst_cnt(cat_instance_count, new_inst_cnt)
                self.img_infos.extend(img_infos)
                ann_infos.append(ann_df)
            self.ann_infos = pandas.concat(ann_infos)
        elif self.dataset_type == 'dense':
            img_infos, img_idx_lookup, new_inst_cnt, ann_df = self.load_annotation_file(ref_file, cat_idx)
            self.img_idx_lookup.update(img_idx_lookup)
            cat_instance_count = self.combine_cat_inst_cnt(cat_instance_count, new_inst_cnt)
            self.img_infos.extend(img_infos)
            self.ann_infos = ann_df

        self.cat_instance_count = cat_instance_count

        # TODO handle complete dataset ann_infos usage

        print("--- ANNOTATION INFO ---")
        print(repr(self))
        print("--- ANNOTATION INFO ---")

        print("loading annotations done in {:.2f}s".format(time() - start_time))

    def generate_annotations(self):
        start = time()
        print("generating annotations...")
        cat_instances = self.cat_instance_count.items()
        new_mapping = None

        cat_with_inst_cnt = sum([1 if cnt > 0 else 0 for _, cnt in cat_instances])
        if cat_with_inst_cnt < len(cat_instances):
            new_mapping = self.get_new_mapping(self.cat_instance_count)
        generate_all_annotations(cat_instances, self.cat_infos, self.root, new_mapping=new_mapping)
        print(f"done generating annotations in {time() - start:.6f}s")

    def check_overlap_train_test(self):
        train_imgs = set()
        for ann_file in tqdm(self.train_ann_files, 'train file checked: ', unit='file'):
            with open(ann_file, 'r') as file:
                data = json.load(file)
            img_fps = [img_info['filename'] for img_info in data['images']]
            train_imgs.update(img_fps)

        test_imgs = set()
        for ann_file in tqdm(self.test_ann_files, 'test file checked: ', unit='file'):
            with open(ann_file, 'r') as file:
                data = json.load(file)
            img_fps = [img_info['filename'] for img_info in data['images']]
            test_imgs.update(img_fps)

        overlap_set = set()
        for file in test_imgs:
            if file in train_imgs:
                overlap_set.add(file)

        print(f"{len(overlap_set)} overlap out of {len(test_imgs)} test and {len(train_imgs)} train images.")
        return overlap_set

    def sort_cat_instances(self):
        cat_cnt = self.cat_instance_count
        self.cat_instance_count = sorted(cat_cnt.items(), key=lambda item: item[1], reverse=True)
        return self.cat_instance_count

    def get_img_infos(self, idxs=None, ids=None):
        self._xor_args(idxs, ids)

        if idxs is not None:
            assert isinstance(idxs, list), 'Given indices idxs must be a ' \
                                           'list or tuple'

            return [self.img_infos[idx] for idx in idxs]
        else:
            assert isinstance(ids, list), 'Given ids must be a list or tuple'
            return [self.img_infos[self.img_idx_lookup[i]] for i in ids]

    def get_anns(self, img_idx=None, img_id=None, ann_set_filter=None):
        self._xor_args(img_idx, img_id)

        if img_idx is not None:
            return self.get_ann_info(self.ann_infos, self.img_infos[img_idx]['ann_ids'],
                                     ann_set_filter)
        else:
            ann_ids = self.img_infos[self.img_idx_lookup[img_id]]['ann_ids']
            return self.get_ann_info(self.ann_infos, ann_ids, ann_set_filter)

    def get_cats(self):
        return {key: value for (key, value) in self.cat_infos.items()
                if value['annotation_set'] in self.chosen_ann_set}

    def get_ann_info(self, ann_infos, ann_ids, ann_set_filter):
        assert isinstance(ann_ids, list), 'Given ann_ids must be a list or tuple'

        ann_ids = [int(i) for i in ann_ids]
        selected = ann_infos.loc[ann_ids]

        # Get annotation set index and return only the specific category id
        if ann_set_filter is None:
            ann_set_filter = self.chosen_ann_set
        if isinstance(ann_set_filter, str):
            ann_set_filter = [ann_set_filter]
        ann_set_idx = [self.annotation_sets.index(ann_set)
                       for ann_set in ann_set_filter]

        def filter_ids(record):
            return [int(record[idx]) for idx in ann_set_idx]

        selected['cat_id'] = selected['cat_id'].map(filter_ids)
        selected = selected[selected['cat_id'].map(lambda x: len(x)) > 0]

        return selected

    # TODO optimize img_infos
    def get_img_ann_pair(self, ann_infos, idxs=None, ids=None, img_infos=None, ann_set_filter=None):
        if img_infos is None:
            self._xor_args(idxs, ids)
            img_infos = self.get_img_infos(idxs, ids)

        annotations = [self.get_ann_info(ann_infos, img_info['ann_ids'], ann_set_filter)
                       for img_info in img_infos]

        return img_infos, annotations

    def _draw_bbox(self, draw, ann, color, oriented, annotation_set=None,
                   print_label=False, print_staff_pos=False, print_onset=False,
                   instances=False):
        annotation_set = 0 if annotation_set is None else annotation_set
        cat_id = ann['cat_id']
        if isinstance(cat_id, list):
            cat_id = int(cat_id[annotation_set])

        parsed_comments = self.parse_comments(ann['comments'])

        if oriented:
            bbox = ann['o_bbox']
            draw.line(bbox + bbox[:2], fill=color, width=3
                      )
        else:
            bbox = ann['a_bbox']
            draw.rectangle(bbox, outline=color, width=2)

        # Now draw the label below the bbox
        x0 = min(bbox[::2])
        y0 = max(bbox[1::2])
        pos = (x0, y0)

        def print_text_label(position, text, color_text, color_box):
            x1, y1 = ImageFont.load_default().getsize(text)
            x1 += position[0] + 4
            y1 += position[1] + 4
            draw.rectangle((position[0], position[1], x1, y1), fill=color_box)
            draw.text((position[0] + 2, position[1] + 2), text, color_text)
            return x1, position[1]

        if instances:
            label = str(int(parsed_comments['instance'].lstrip('#'), 16))
            print_text_label(pos, label, '#ffffff', '#303030')

        else:
            label = self.cat_infos[cat_id]['name']

            if print_label:
                pos = print_text_label(pos, label, '#ffffff', '#303030')
            if print_onset and 'onset' in parsed_comments.keys():
                pos = print_text_label(pos, parsed_comments['onset'], '#ffffff',
                                       '#091e94')
            if print_staff_pos and 'rel_position' in parsed_comments.keys():
                print_text_label(pos, parsed_comments['rel_position'],
                                 '#ffffff', '#0a7313')

        return draw

    def visualize(self,
                  img_idx=None,
                  img_id=None,
                  out_dir=None,
                  annotation_set=None,
                  oriented=True,
                  instances=False,
                  show=True):
        # Since we can only visualize a single image at a time, we do i[0] so
        # that we don't have to deal with lists. get_img_ann_pair() returns a
        # tuple that's why we use list comprehension
        img_idx = [img_idx] if img_idx is not None else None
        img_id = [img_id] if img_id is not None else None

        if annotation_set is None:
            annotation_set = 0
            self.chosen_ann_set = self.annotation_sets[0]
        else:
            annotation_set = self.annotation_sets.index(annotation_set)
            self.chosen_ann_set = self.chosen_ann_set[annotation_set]

        img_info, ann_info = [i[0] for i in
                              self.get_img_ann_pair(self.ann_infos,
                                                    idxs=img_idx, ids=img_id)]

        img_dir = path.join(self.root, 'images')
        seg_dir = path.join(self.root, 'segmentation')
        inst_dir = path.join(self.root, 'instance')

        # Get the actual image filepath and the segmentation filepath
        img_fp = path.join(img_dir, img_info['filename'])
        print(f'Visualizing {img_fp}...')

        # Remember: PIL Images are in form (h, w, 3)
        img = Image.open(img_fp)

        if instances:
            # Do stuff
            inst_fp = path.join(
                inst_dir,
                path.splitext(img_info['filename'])[0] + '_inst.png'
            )
            overlay = Image.open(inst_fp)
            img.putalpha(255)
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')

        else:
            seg_fp = path.join(
                seg_dir,
                path.splitext(img_info['filename'])[0] + '_seg.png'
            )
            overlay = Image.open(seg_fp)

            # Here we overlay the segmentation on the original image using the
            # colorcet colors
            # First we need to get the new color values from colorcet
            colors = [ImageColor.getrgb(i) for i in colorcet.glasbey]
            colors = numpy.array(colors).reshape(768, ).tolist()
            colors[0:3] = [0, 0, 0]  # Set background to black

            # Then put the palette
            overlay.putpalette(colors)
            overlay_array = numpy.array(overlay)

            # Now the img and the segmentation can be composed together. Black
            # areas in the segmentation (i.e. background) are ignored

            mask = numpy.zeros_like(overlay_array)
            mask[numpy.where(overlay_array == 0)] = 255
            mask = Image.fromarray(mask, mode='L')

            img = Image.composite(img, overlay.convert('RGB'), mask)
        draw = ImageDraw.Draw(img)

        # Now draw the gt bounding boxes onto the image
        for ann in ann_info.to_dict('records'):
            draw = self._draw_bbox(draw, ann, '#ed0707', oriented,
                                   annotation_set, instances)

        if show:
            img.show()
        if out_dir is not None:
            img.save(path.join(out_dir, datetime.now().strftime('%m-%d_%H%M%S'))
                     + '.png')

    # def visualize_categories(self):
    #     self.cat_info

    def remap_empty_cat_id(self, anns, new_mapping):
        print("remapping...")
        start_time = time()
        
        remap_cnt = 0
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in new_mapping:
                # print(f"remapping {cat_id} to {new_mapping[cat_id]}")
                ann['category_id'] = new_mapping[cat_id]
                remap_cnt += 1

        print(f"remapped {remap_cnt} annotations in t={time() - start_time:.6f}s")
        return new_mapping, anns

    def get_new_mapping(self, cat_instances, starts_from=0):
        new_mapping = dict()
        print("get new mappings...")
        cat_len = len(self.get_cats())

        end = len(cat_instances) + starts_from
        print(f"start to end: {starts_from, end}")
        for i in range(starts_from, end):
            if i == cat_len:
                print(f"i reached cat_len {i, cat_len}.")
                if i not in cat_instances or cat_instances[i] == 0:
                    print(f"hey! class id {i} is empty too!")
                    cat_len -= 1
                break

            if i not in cat_instances or cat_instances[i] == 0:
                while cat_len not in cat_instances or cat_instances[cat_len] == 0:
                    cat_len -= 1
                    if cat_len <= i:
                        print(f"cat len {cat_len} reached i ({i})!")
                        break
                new_mapping[cat_len] = i
                cat_len -= 1
        print(f"new mapping: {new_mapping}")
        return new_mapping

    def crop_image_objects_complete(self,
                                    executor,
                                    img,
                                    ann_info,
                                    class_counter,
                                    out_dir,
                                    bg_opacity=0,
                                    min_dim=150,
                                    resize=None):
        # split the gt bounding boxes onto the image
        for ann in ann_info.to_dict('records'):
            cat_id = ann['cat_id'][0]
            name = self.cat_infos[cat_id]['name']
            class_counter[cat_id] += 1

            out_fp = path.join(out_dir, f'{name}-{class_counter[cat_id]:08d}') + '.png'
            bbox = self.moderate_bbox(ann['a_bbox'], img.size)
            executor.submit(self.crop_save_obj_img, img, bbox, resize, min_dim, bg_opacity, out_fp, debug)
        return

    def crop_image_objects_dense(self, img, ann_info, class_counter, out_dir, bg_opacity=0, min_dim=150, resize=None):
        # split the gt bounding boxes onto the image
        with futures.ThreadPoolExecutor() as executor:
            for ann in ann_info.to_dict('records'):

                cat_id = ann['cat_id'][0]
                name = self.cat_infos[cat_id]['name']
                class_counter.setdefault(cat_id, 0)
                class_counter[cat_id] += 1

                bbox = self.moderate_bbox(ann['a_bbox'], img.size)
                out_fp = path.join(out_dir, f'{name}-{class_counter[cat_id]:06d}') + '.png'

                executor.submit(self.crop_save_obj_img, img, bbox, resize, min_dim, bg_opacity, out_fp)

        return class_counter

    def crop_all_to_instances(self,
                              out_dir=None,
                              annotation_set=None,
                              bg_opacity=0,
                              resize=None,
                              cat_inst_ctr=None,
                              cont=True,
                              sav=None):
        assert out_dir is not None, "out_dir can't be empty"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if cat_inst_ctr is None:
            cat_inst_ctr = dict()
            for cat in self.get_cats():
                cat_inst_ctr.setdefault(cat, 0)

        if annotation_set is None:
            self.chosen_ann_set = self.annotation_sets[0]
        else:
            self.chosen_ann_set = self.chosen_ann_set[annotation_set]

        print("initiate cropping...")
        start_time = time()

        if self.dataset_type == 'complete':
            fps = self.train_ann_files + self.test_ann_files
            cat_idx = self.annotation_sets.index(self.chosen_ann_set)
            for file in tqdm(fps, 'read file', unit='file'):
                with futures.ThreadPoolExecutor() as executor:
                    img_infos, _, new_inst_cnt, ann_df = self.load_annotation_file(file, cat_idx)
                    for img_info in tqdm(img_infos, 'queued imgs', unit='img'):
                        # for img_info in img_infos:
                        _, ann_info = [i[0] for i in self.get_img_ann_pair(ann_df, img_infos=[img_info])]
                        img_fp = path.join(self.root, 'images', img_info['filename'])
                        img = Image.open(img_fp)

                        self.crop_image_objects_complete(executor, img, ann_info, cat_inst_ctr, out_dir, bg_opacity,
                                                         resize=resize)
                    cat_inst_ctr = self.combine_cat_inst_cnt(cat_inst_ctr, new_inst_cnt)
            print(f'done t={time() - start_time:.6f}s')
        elif self.dataset_type == 'dense':
            skip = False
            prog_file = sav if sav is not None else path.join(self.root, 'crop_dense_prog.json')
            if cont:
                if path.isfile(prog_file):
                    skip = True
                    with open(prog_file, 'r') as file:
                        data = json.load(file)
                        last_file = data['last_file']
                        for k, cnt in data['cat_inst_ctr'].items():
                            cat_inst_ctr[int(k)] = cnt
                        print(f"cat_inst_cnt: {cat_inst_ctr}")
                        print("continuing..")
            for img_info in tqdm(self.img_infos, desc='progress', unit='img'):
                if skip:
                    if img_info['filename'] == last_file:
                        skip = False
                    continue

                img_fp = path.join(self.root, 'images', img_info['filename'])
                _, ann_info = [i[0] for i in self.get_img_ann_pair(self.ann_infos, img_infos=[img_info])]
                img = Image.open(img_fp)

                cat_inst_ctr = self.crop_image_objects_dense(img, ann_info, cat_inst_ctr, out_dir, bg_opacity, resize=resize)

                with open(prog_file, 'w') as file:
                    json.dump({'last_file': img_info['filename'], 'cat_inst_ctr': cat_inst_ctr}, file)
            print(f"cats: {cat_inst_ctr}")
            print(f'done t={time() - start_time:.6f}s')
            # print(f"total annotation {len(annotations)} with {len(cat_inst_ctr)} out of {num_classes} classes")
        return cat_inst_ctr
