import json
from datetime import datetime
from typing import List
from os import path
from time import time

from PIL import Image, ImageColor, ImageDraw, ImageFont
import colorcet
import numpy
import pandas


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
                  data_root=None,
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

        # Get the data_root from the ann_file path if it doesn't exist
        if data_root is None:
            data_root = path.split(self.train_ann_file)[0]

        img_dir = path.join(data_root, 'images')
        seg_dir = path.join(data_root, 'segmentation')
        inst_dir = path.join(data_root, 'instance')

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

    def crop_all_to_instances(self):
        # TODO
        return
