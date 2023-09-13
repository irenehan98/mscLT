"""
partially based on obb_anns (https://github.com/yvan674/obb_anns)
"""
import json
import math
import random
from os import path
from time import time

from PIL import Image


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
        idx_set.add(random.randint(1, rand_max))
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
        for idx in range(1, cnt+1):
            if idx in excl_sets[cat]:
                continue
            # WARNING: quick implementation only, use your own params!
            width, height = (150, 150)
            # WARNING: len depends on dataset_type
            img_fp = path.join(img_dir, f'{name}-{idx:08d}') + '.png'
            if style == 'bbn':
                annotations.append(new_bbn_ann_inst(cat_id, current_id, height, img_fp, width))
            current_id += 1
    with open(path.join(home_dir, filename), 'w') as fp:
        json.dump({'annotations': annotations, 'num_classes': cat_cnt, "remapped_cats": new_mapping}, fp)
    return


def generate_annotation(idx_sets, cat_info, new_mapping, img_dir, filename, current_id, style):
    print(f"generating {filename}..")
    annotations = []
    for cat, idx_set in idx_sets:
        name = cat_info[cat]['name']
        # TODO handle new_mapping value None
        # cat_id = cat if cat not in new_mapping else new_mapping[cat]
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
                    "category_id": cat,
                    "fpath": img_fp,
                })
            current_id += 1
    for ann in annotations:
        if ann['category_id'] in new_mapping:
            ann['category_id'] = new_mapping[ann['category_id']]
    with open(path.join(img_dir, filename), 'w') as fp:
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
def generate_all_annotations(cat_instances, cat_info, img_dir, style='bbn', new_mapping=None):
    test_set, val_set = generate_test_val_set(cat_instances, 0.2, 0.2)

    current_id = 1
    current_id = generate_annotation(test_set.items(), cat_info, new_mapping, img_dir, 'test.json', current_id, style)
    current_id = generate_annotation(val_set.items(), cat_info, new_mapping, img_dir, 'val.json', current_id, style)

    combined_set = test_set
    for k, v in val_set.items():
        combined_set[k].update(v)
    generate_train_annotations(cat_instances, combined_set, cat_info, new_mapping, img_dir, 'train.json', current_id, style)
    return


def convert_bbn_to_ride(home_dir, json_f, txt):
    print(f"converting {json_f}")
    start = time()
    with open(path.join(home_dir, json_f), 'r') as ann_file:
        annotations = json.load(ann_file)['annotations']
    f = ""
    for ann in annotations:
        img_path = ann['fpath']
        label = ann['category_id']
        f += f"{img_path} {label}\n"
    with open(path.join(home_dir, txt), 'w') as fp:
        fp.write(f)
    print(f"convert to {txt} done: {time() - start:.6f}s")


# TODO rename to crop_with_bg
def crop_object(img, bbox, min_dim, bg_opacity, resize):
    x0, y0, x1, y1 = bbox
    width, height = int(x1 - x0), int(y1 - y0)
    max_width, max_height = img.size

    bg_wdiff, bg_hdiff = max(0, min_dim - width), max(0, min_dim - height)
    bg_x0, bg_x1 = max(0, x0 - (bg_wdiff / 2)), min(max_width, x1 + (bg_wdiff / 2))
    bg_y0, bg_y1 = max(0, y0 - (bg_hdiff / 2)), min(max_height, y1 + (bg_hdiff / 2))
    bg_width, bg_height = int(bg_x1-bg_x0), int(bg_y1-bg_y0)

    cropped = img.crop((x0, y0, x1, y1))

    result = Image.new(cropped.mode, (bg_width, bg_height), (255, 255, 255))
    if bg_opacity and bg_opacity > 0:
        bg_cropped = img.crop((bg_x0, bg_y0, bg_x1, bg_y1))
        bg_cropped.putalpha(bg_opacity)
        result.paste(bg_cropped, (0, 0), bg_cropped)
    result.paste(cropped, (int(x0 - bg_x0), int(y0 - bg_y0)))

    if resize is not None:
        result = result.resize(resize)

    return result


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
