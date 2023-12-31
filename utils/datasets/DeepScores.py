import itertools
from concurrent import futures
from datetime import datetime
from enum import Enum
from os import mkdir
from time import time
from typing import List

import colorcet
import numpy
import pandas
from PIL import ImageDraw, ImageFont, ImageColor
from tqdm import tqdm

from ..dataset import *


class DeepScoresType(Enum):
    DENSE = 'dense'
    COMPLETE = 'complete'

    @staticmethod
    def from_str(label):
        if label in ('dense', 'DENSE'):
            return DeepScoresType.DENSE
        elif label in ('complete', 'COMPLETE'):
            return DeepScoresType.COMPLETE
        else:
            raise NotImplementedError


class DeepScores:
    def __init__(self, root_dir, dataset_type, out_dir=None):
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

        self.dataset_type = DeepScoresType.from_str(dataset_type)
        self.home_dir = path.join(root_dir, 'ds2_' + self.dataset_type.value)
        self.out_dir = out_dir if out_dir is not None else self.home_dir

        self.train_ann_fps = []
        self.test_ann_fps = []
        if self.dataset_type == DeepScoresType.DENSE:
            self.img_base = '{}-{:06d}.png'
            self.train_ann_fps.append(path.join(self.home_dir, "deepscores_train.json"))  # default ann_file
            self.test_ann_fps.append(path.join(self.home_dir, "deepscores_test.json"))
        elif self.dataset_type == DeepScoresType.COMPLETE:
            self.img_base = '{}-{:08d}.png'
            for i in itertools.count():
                test_fp = path.join(self.home_dir, f"deepscores-complete-{i}_test.json")
                train_fp = path.join(self.home_dir, f"deepscores-complete-{i}_train.json")

                if path.isfile(train_fp):
                    self.train_ann_fps.append(train_fp)
                if path.isfile(test_fp):
                    self.test_ann_fps.append(test_fp)

                if not path.isfile(train_fp) and not path.isfile(test_fp):
                    break

        print("initialization done in {:.6f}s".format(time() - start_time))

    def __repr__(self):
        information = "DeepScore Dataset Annotation\n"
        information += f"root file: {self.home_dir}\n"
        information += f"dataset type: {self.dataset_type}\n"
        if self.dataset_info is not None:
            information += f"Num images: {len(self.img_infos)}\n"
            information += f"Num anns: {len(self.ann_infos) if self.dataset_type == DeepScoresType.DENSE else '-'}\n"
            information += f"Num cats: {len(self.cat_infos)}\n"
            information += f"cat insts: {self.cat_instance_count}\n"
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

    def _gen_inst_ctr(self):
        cat_instance_count = dict()
        for cat in self.get_cats():
            cat_instance_count.setdefault(cat, 0)
        return cat_instance_count

    @staticmethod
    def parse_comments(comment):
        parsed_dict = dict()
        for co in comment.split(";"):
            if len(co.split(":")) > 1:
                key, value = co.split(":")
                parsed_dict[key] = value
        return parsed_dict

    @staticmethod
    def combine_cat_inst_cnt(inst_cnt, more_cnt):
        # WARNING: unknown categories in muscima++ has None as key values
        for k, v in more_cnt.items():
            inst_cnt[int(k)] += v
        return inst_cnt

    @staticmethod
    def count_empty_cat_cnt(cat_instances):
        return sum([1 if cnt == 0 else 0 for _, cnt in cat_instances])

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

    @staticmethod
    def remap_empty_cat_id(anns, new_mapping):
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

    def generate_img_name(self, name, inst_cnt):
        return self.img_base.format(name, inst_cnt)

    def load_annotations(self, annotation_set_filter=None, load_all=True):
        print("loading annotations...")
        start_time = time()

        # -- only need to load once --
        ref_fp = self.test_ann_fps[0]

        with open(ref_fp, 'r') as ann_fp:
            data = json.load(ann_fp)

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

        print(f"basic info loaded in {time() - start_time:.6f}s..")
        # -- end of load once --
        cat_instance_count = self._gen_inst_ctr()

        self.img_infos = []

        # press F for my 32gb ram it ain't enough for multithreading
        if load_all:
            with_ann_df = True if self.dataset_type == DeepScoresType.DENSE else False
            fps = self.train_ann_fps + self.test_ann_fps
            ann_infos = []
            for ann_fp in tqdm(fps, 'file processed: ', unit='file'):
                img_infos, img_idx_lookup, new_inst_cnt, ann_df = self.load_annotation_file(ann_fp, cat_idx,
                                                                                            with_ann_df)
                self.img_idx_lookup.update(img_idx_lookup)
                cat_instance_count = self.combine_cat_inst_cnt(cat_instance_count, new_inst_cnt)
                self.img_infos.extend(img_infos)
                ann_infos.append(ann_df)
            self.ann_infos = pandas.concat(ann_infos)
        elif self.dataset_type == DeepScoresType.DENSE:
            img_infos, img_idx_lookup, new_inst_cnt, ann_df = self.load_annotation_file(ref_fp, cat_idx)
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

    def generate_annotations(self, img_dir):
        start = time()
        print("generating annotations...")
        img_dir = path.join(self.out_dir, img_dir)
        assert path.isdir(img_dir), f"path {img_dir} does not exist"

        cat_instances = self.cat_instance_count.items()
        print(cat_instances)
        # TODO can bug if model requires cat to start at 0 (DeepScores starts at 1)
        empty_cat_cnt = self.count_empty_cat_cnt(cat_instances)
        new_mapping = self.get_new_mapping(self.cat_instance_count) if empty_cat_cnt > 0 else None

        generate_all_annotations(cat_instances, self.cat_infos, img_dir, new_mapping=new_mapping)
        print(f"done generating annotations in {time() - start:.6f}s")

    def check_overlap_train_test(self):
        train_imgs = set()
        for ann_fp in tqdm(self.train_ann_fps, 'train file checked: ', unit='file'):
            with open(ann_fp, 'r') as file:
                data = json.load(file)
            img_fps = [img_info['filename'] for img_info in data['images']]
            train_imgs.update(img_fps)

        test_imgs = set()
        for ann_fp in tqdm(self.test_ann_fps, 'test file checked: ', unit='file'):
            with open(ann_fp, 'r') as file:
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
            draw.line(bbox + bbox[:2], fill=color, width=3)
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

        img_dir = path.join(self.home_dir, 'images')
        seg_dir = path.join(self.home_dir, 'segmentation')
        inst_dir = path.join(self.home_dir, 'instance')

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

    # TODO
    # def visualize_categories(self):
    #     self.cat_info

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
                    print(f"hej class id {i} is empty too")
                    cat_len -= 1
                break

            if i not in cat_instances or cat_instances[i] == 0:
                while cat_len not in cat_instances or cat_instances[cat_len] == 0:
                    cat_len -= 1
                    if cat_len <= i:
                        print(f"cat len {cat_len} reached i ({i})")
                        break
                new_mapping[cat_len] = i
                cat_len -= 1
        print(f"new mapping: {new_mapping}")
        return new_mapping

    def crop_image_objects_threaded(self,
                                    executor,
                                    img,
                                    ann_info,
                                    class_counter,
                                    out_dir,
                                    bg_opacity=None,
                                    min_dim=150,
                                    resize=None):
        # split the gt bounding boxes onto the image
        for ann in ann_info.to_dict('records'):
            cat_id = ann['cat_id'][0]
            name = self.cat_infos[cat_id]['name']
            class_counter[cat_id] += 1

            out_fp = path.join(out_dir, self.generate_img_name(name, class_counter[cat_id]))
            bbox = moderate_bbox(ann['a_bbox'], img.size)
            executor.submit(self.crop_save_obj_img, img, bbox, resize, min_dim, bg_opacity, out_fp)
        return

    def crop_image_objects(self, img, ann_info, class_counter, out_dir, bg_opacity=None, min_dim=150, resize=None):
        # split the gt bounding boxes onto the image
        for ann in ann_info.to_dict('records'):
            cat_id = ann['cat_id'][0]
            name = self.cat_infos[cat_id]['name']
            class_counter.setdefault(cat_id, 0)
            class_counter[cat_id] += 1

            bbox = moderate_bbox(ann['a_bbox'], img.size)
            out_fp = path.join(out_dir, self.generate_img_name(name, class_counter[cat_id]))

            self.crop_save_obj_img(img, bbox, resize, min_dim, bg_opacity, out_fp)

        return class_counter

    def crop_all_to_instances(self,
                              out_dir,
                              annotation_set=None,
                              bg_opacity=None,
                              resize=None,
                              cont=True,
                              sav=None):
        assert out_dir is not None, "out_dir can't be empty"
        crop_out = path.join(self.out_dir, out_dir)
        if not path.exists(crop_out):
            mkdir(crop_out)
        self.crop_out = crop_out

        cat_inst_ctr = self._gen_inst_ctr()

        # defaults DeepScores
        self.chosen_ann_set = self.annotation_sets[0 if annotation_set is None else annotation_set]

        print("initiate cropping...")
        start_time = time()

        skip = False
        prog_fp = path.join(self.home_dir, sav if sav is not None else 'crop_prog.json')
        if cont and path.isfile(prog_fp):
            skip = True
            with open(prog_fp, 'r') as file:
                data = json.load(file)
                last_file = data['last_file']
                for k, cnt in data['cat_inst_ctr'].items():
                    cat_inst_ctr[int(k)] = cnt
                print(f"cat_inst_cnt: {cat_inst_ctr}")
                print("continuing..")

        fps = self.train_ann_fps + self.test_ann_fps
        cat_idx = self.annotation_sets.index(self.chosen_ann_set)
        for file in tqdm(fps, 'read file', unit='file'):
            img_infos, _, new_inst_cnt, ann_df = self.load_annotation_file(file, cat_idx)
            for img_info in tqdm(img_infos, 'progress', unit='img'):
                if skip:
                    if img_info['filename'] == last_file:
                        skip = False
                    continue

                _, ann_info = [i[0] for i in self.get_img_ann_pair(ann_df, img_infos=[img_info])]
                img_fp = path.join(self.home_dir, 'images', img_info['filename'])
                img = Image.open(img_fp)

                cat_inst_ctr = self.crop_image_objects(img, ann_info, cat_inst_ctr, crop_out, bg_opacity,
                                                       resize=resize)
                with open(prog_fp, 'w') as file:
                    json.dump({'last_file': img_info['filename'], 'cat_inst_ctr': cat_inst_ctr}, file)

        self.assert_crop_success(out_dir, cat_inst_ctr)

        self.cat_instance_count = dict()
        crop_file = dict()
        for cat, cnt in sorted(cat_inst_ctr.items(), key=lambda item: item[1], reverse=True):
            self.cat_instance_count[cat] = cnt
            crop_file[self.cat_infos[cat]['name']] = cnt

        with open(path.join(self.home_dir, "cats.json"), 'w') as f:
            json.dump({'cats': crop_file}, f)
        print(f'done t={time() - start_time:.6f}s')
        # print(f"total annotation {len(annotations)} with {len(cat_inst_ctr)} out of {num_classes} classes")
        return crop_file, cat_inst_ctr

    def assert_crop_success(self, cat_instances, out_dir):
        not_found = self._gen_inst_ctr()

        for cat, cnt in cat_instances.items():
            for i in range(1, cnt + 1):
                name = self.cat_infos[cat]['name']
                if not path.isfile(path.join(self.out_dir, out_dir, self.generate_img_name(name, cnt))):
                    not_found[cat] += 1

        print(f"non-existing image count: {not_found}")
        if sum([1 if cnt > 0 else 0 for _, cnt in cat_instances]) > 0:
            print("there are incomplete crops!")
