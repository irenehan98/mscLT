"""
partially based on obb_anns (https://github.com/yvan674/obb_anns)
"""

import json
from datetime import datetime
from typing import List
from os import path
from time import time

from PIL import Image, ImageColor, ImageDraw, ImageFont
import colorcet
import numpy
import pandas


def crop_object(img, bbox, min_dim, bg_opacity, resize, debug):
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

    if debug is True:
        print(f"wdiff: {bg_wdiff} hdiff: {bg_hdiff}")
        print(f"bg: {bg_x0, bg_y0, bg_x1, bg_y1}")
        print(f"cropped: {cropped.size} bg: {bg_cropped.size}")
        cropped.show()
        bg_cropped.show()
        print(f"result: {result.size}")

    return result


class DeepScores:
    def __init__(self, root_dir, dataset_type):
        print("initializing...")
        start_time = time()

        self.dataset_info = None
        self.annotation_sets = None
        self.chosen_ann_set = None  # type: None or List[str]
        self.cat_instance_count = dict()
        self.ann_info = None
        self.cat_info = None
        self.img_info = None  # img_info has 'id' 'filename' 'width' 'height' 'ann_ids'
        self.img_idx_lookup = dict()
        self.ann_ids = []

        self.root = root_dir
        # TODO: assert dataset exists
        # TODO: find out if deepscores train and test data overlaps
        if dataset_type == 'dense':
            self.train_ann_file = path.join(root_dir, "deepscores_train.json")  # default ann_file
            self.test_ann_file = path.join(root_dir, "deepscores_test.json")
        else:
            print(f"unsupported type: {dataset_type}")
            return

        print("initialization done in {:.6f}s".format(time() - start_time))

    def __repr__(self):
        information = "DeepScore Dataset Annotation\n"
        information += f"root file: {self.root}\n"
        if self.dataset_info is not None:
            information += f"Num images: {len(self.img_info)}\n"
            information += f"Num anns: {len(self.ann_info)}\n"
            information += f"Num cats: {len(self.cat_info)}"
        else:
            information += "Annotations not yet loaded\n"
        return information

    def __len__(self):
        return 0 if self.img_info is None else len(self.img_info)

    @staticmethod
    def _xor_args(m, n):
        only_one_arg = ((m is not None and n is None)
                        or (m is None and n is not None))
        assert only_one_arg, 'Only one type of request can be done at a time'

    @staticmethod
    def parse_comments(comment):
        """Parses the comment field of an annotation.

        :returns dictionary with every comment name as keys
        :rtype: dict
        """
        parsed_dict = dict()
        for co in comment.split(";"):
            if len(co.split(":")) > 1:
                key, value = co.split(":")
                parsed_dict[key] = value
        return parsed_dict

    def load_annotations(self, annotation_set_filter=None):
        print("loading annotations...")
        start_time = time()

        with open(self.train_ann_file, 'r') as train_ann_file:
            data = json.load(train_ann_file)

        self.dataset_info = data['info']
        self.annotation_sets = data['annotation_sets']

        # Sets annotation sets and makes sure it exists
        if annotation_set_filter is not None:
            assert annotation_set_filter in self.annotation_sets, \
                f"The chosen annotation_set_filter " \
                f"{annotation_set_filter} is not a in the available " \
                f"annotations sets."
            self.chosen_ann_set = annotation_set_filter
            annotation_idx = self.annotation_sets.index(annotation_set_filter)
        else:
            self.chosen_ann_set = self.annotation_sets
            annotation_idx = 0

        self.cat_info = {int(k): v for k, v in data['categories'].items()}

        # Process annotations
        cat_inst_count = dict()

        ann_id = []
        annotations = {'a_bbox': [],
                       'o_bbox': [],
                       'cat_id': [],
                       'area': [],
                       'img_id': [],
                       'comments': []}

        for k, v in data['annotations'].items():
            # TODO find better alternative (from cat_info?)
            cat_id = v['cat_id'][annotation_idx]
            cat_inst_count.setdefault(cat_id, 0)
            cat_inst_count[cat_id] += 1

            ann_id.append(int(k))
            annotations['a_bbox'].append(v['a_bbox'])
            annotations['o_bbox'].append(v['o_bbox'])
            annotations['cat_id'].append(v['cat_id'])
            annotations['area'].append(v['area'])
            annotations['img_id'].append(v['img_id'])
            annotations['comments'].append(v['comments'])

        class_keys = list(cat_inst_count.keys())
        self.cat_instance_count = {i: cat_inst_count[i] for i in class_keys}

        if self.ann_info is None:
            self.ann_info = pandas.DataFrame(annotations, ann_id)
        # else:
        #     # TODO support complete version

        if self.img_info is None:
            self.img_info = data['images']
        # else:
        #     # TODO support complete version
        #     self.img_info.extend(data['images'])

        for i, img in enumerate(data['images']):
            self.ann_ids.extend(img['ann_ids'])
            # lookup table used to figure out the index in img_info of image based on their img_id
            self.img_idx_lookup[int(img['id'])] = i

        print("--- ANNOTATION INFO ---")
        print(repr(self))
        print("--- ANNOTATION INFO ---")

        print("loading annotations done in {:.2f}s".format(time() - start_time))

    def get_imgs(self, idxs=None, ids=None):
        self._xor_args(idxs, ids)

        if idxs is not None:
            assert isinstance(idxs, list), 'Given indices idxs must be a ' \
                                           'list or tuple'

            return [self.img_info[idx] for idx in idxs]
        else:
            assert isinstance(ids, list), 'Given ids must be a list or tuple'
            return [self.img_info[self.img_idx_lookup[i]] for i in ids]

    def get_anns(self, img_idx=None, img_id=None, ann_set_filter=None):
        self._xor_args(img_idx, img_id)

        if img_idx is not None:
            return self.get_ann_info(self.img_info[img_idx]['ann_ids'],
                                     ann_set_filter)
        else:
            ann_ids = self.img_info[self.img_idx_lookup[img_id]]['ann_ids']
            return self.get_ann_info(ann_ids, ann_set_filter)

    def get_cats(self):
        return {key: value for (key, value) in self.cat_info.items()
                if value['annotation_set'] in self.chosen_ann_set}

    def get_ann_info(self, ann_ids, ann_set_filter=None):
        assert isinstance(ann_ids, list), 'Given ann_ids must be a list or tuple'

        ann_ids = [int(i) for i in ann_ids]
        selected = self.ann_info.loc[ann_ids]

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

    def get_img_ann_pair(self, idxs=None, ids=None, ann_set_filter=None):
        self._xor_args(idxs, ids)

        images = self.get_imgs(idxs, ids)
        annotations = [self.get_ann_info(img['ann_ids'], ann_set_filter)
                       for img in images]

        return images, annotations

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
            label = self.cat_info[cat_id]['name']

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
                              self.get_img_ann_pair(
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

    # TODO evaluate if redundant
    def count_cats(self, annotation_set=None):
        if annotation_set != 'deepscores' and annotation_set != 'muscima++':
            return None
        counter = 0
        for v in self.cat_info.values():
            if v['annotation_set'] == annotation_set:
                counter += 1
        return counter

    def remap_empty_cat_id(self, anns, class_counter, new_mapping=None, starts_from_one=False):
        rev_cat_ctr = None
        if new_mapping is None:
            new_mapping = dict()

        print("remapping categories...")
        start_time = time()

        if len(new_mapping) <= 0:
            rev_cat_ctr = self.count_cats(self.chosen_ann_set)

            start = 1 if starts_from_one else 0
            end = len(class_counter) + start

            for i in range(start, end):
                # print(f"[debug] i: {i}")
                if i is rev_cat_ctr:
                    print(f"[debug] i, rev_cat_ctr: {i, rev_cat_ctr}")
                    if i not in class_counter or class_counter[i] == 0:
                        print(f"class id {i} is empty")
                        rev_cat_ctr -= 1
                    break

                if i not in class_counter or class_counter[i] == 0:
                    print(f"key {i} not in class_counter; rev_cat_ctr: {rev_cat_ctr, rev_cat_ctr in class_counter}")
                    while rev_cat_ctr not in class_counter or class_counter[rev_cat_ctr] == 0:
                        rev_cat_ctr -= 1
                        # print(f"[debug] back_ctr: {rev_cat_ctr}")
                    print(f"key {rev_cat_ctr} is remapped to {i}")
                    new_mapping[rev_cat_ctr] = i
                    rev_cat_ctr -= 1
            print(f"rev_cat_ctr: {rev_cat_ctr}")
            print(f"new mapping: {new_mapping}")

        remap_cnt = 0
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in new_mapping:
                # print(f"remapping {cat_id} to {new_mapping[cat_id]}")
                ann['category_id'] = new_mapping[cat_id]
                remap_cnt += 1

        print(f"remapped {remap_cnt} annotations in t={time() - start_time}s")
        return rev_cat_ctr, new_mapping, anns

    def crop_bounding_boxes(self,
                            img_id,
                            class_counter,
                            out_dir,
                            annotation_set=None,
                            last_id=0,
                            bg_opacity=0,
                            min_dim=150,
                            resize=None,
                            ann_style=None,
                            verbose=False,
                            debug=False):

        if annotation_set is None:
            self.chosen_ann_set = self.annotation_sets[0]
        else:
            self.chosen_ann_set = self.chosen_ann_set[annotation_set]

        img_info, ann_info = [i[0] for i in
                              self.get_img_ann_pair(ids=img_id)]

        img_dir = path.join(self.root, 'images')
        img_fp = path.join(img_dir, img_info['filename'])

        # Remember: PIL Images are in form (h, w, 3)
        img = Image.open(img_fp)
        img_width, img_height = img.size

        new_anns = []
        # split the gt bounding boxes onto the image
        for ann in ann_info.to_dict('records'):
            x0, y0, x1, y1 = ann['a_bbox']

            if x0 == x1:
                x0 = max(0, x0 - 1)
                x1 = min(img_width, x1 + 1)
            if y0 == y1:
                y0 = max(0, y0 - 1)
                y1 = min(img_height, y1 + 1)

            bbox = [x0, y0, x1, y1]

            last_id += 1
            current_id = last_id

            cat_id = ann['cat_id'][0]
            name = self.cat_info[cat_id]['name']
            class_counter.setdefault(cat_id, 0)
            class_counter[cat_id] += 1

            result = crop_object(img, bbox, min_dim, bg_opacity, resize, debug)

            if verbose:
                width, height = result.size
                if width > min_dim or height > min_dim:
                    print(f"[WARNING] obj {name} has dimension {result.size}")

            out_fp = path.join(out_dir, f'{name}-{class_counter[cat_id]:06d}') + '.png'
            result.save(out_fp)

            if ann_style == "BBN":
                width, height = result.size
                new_anns.append({
                    "image_id": current_id,
                    "im_height": height,
                    "im_width": width,
                    "category_id": cat_id,
                    "fpath": out_fp,
                })

            if debug is True:
                result.show()
                break

        if debug:
            print(new_anns)

        return last_id, new_anns

    def crop_all_to_instances(self,
                              out_dir=None,  # TODO can't be none
                              out_annot='annotations.json',
                              annotation_set=None,
                              bg_opacity=0,
                              ann_style=None,
                              resize=None,
                              verbose=False,
                              debug=False,
                              class_counter=None,
                              last_id=0):
        # TODO multithreading

        if class_counter is None:
            class_counter = dict()

        print("initiate cropping...")
        start_time = time()

        annotations = []

        # # TODO make progress visualizer
        # progress_ctr = 0
        for img_info in self.img_info:
            # if progress_ctr % int(len(self.img_info) / 10) == 0:
            #     print(f"progress: {int(progress_ctr / len(self.img_info) * 100)}%")
            # progress_ctr += 1

            img_id = [img_info['id']]
            last_id, bb_annotations = self.crop_bounding_boxes(img_id, class_counter, out_dir, annotation_set, last_id, bg_opacity, resize=resize, ann_style=ann_style, verbose=verbose, debug=debug)
            annotations.extend(bb_annotations)

            if debug:
                print(annotations)
                break

        class_keys = list(class_counter.keys())
        class_keys.sort()
        class_counter = {i: class_counter[i] for i in class_keys}
        print(f"class_counter: {class_counter}")

        num_classes = self.count_cats(self.chosen_ann_set)

        if ann_style == "BBN":
            if len(class_counter) < num_classes:
                num_classes, remapped_cat, annotations = self.remap_empty_cat_id(annotations, class_counter)

        # turn into json
        with open(path.join(self.root, out_annot), 'w') as fp:
            json.dump({'annotations': annotations, 'num_classes': num_classes, "remapped_cat_id": remapped_cat}, fp)

        print(f'done t={time() - start_time}s: total annotation {len(annotations)} with {len(class_counter)} out of {num_classes} classes')
        return last_id, class_counter
