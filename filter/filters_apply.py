import os
from tqdm import tqdm 
from PIL import Image

from util import Time, open_image_np, np_to_pil, mask_percent
from color_filter import color_filter, color_filter_v2


def apply_color_filter_to_image(item, src, fitler_ing_dir,  filter_dir, filter_mask_dir, filter_cmp_dir, filter_mask_cmp_dir, 
                                save=True, display=False, hole_size=1000):
    t = Time()
    print('Processing slide:  {}'.format(item))
    
    image_path = os.path.join(src, item)
    # np_orig = open_pad_image_np(image_path)
    np_orig = open_image_np(image_path)
    item = item.split('.')[0]
    filtered_np_img, filtered_mask, filter_np_img_cmp, filtered_mask_cmp = color_filter(np_orig, item, save_dir=fitler_ing_dir, save=False, display=display, hole_size=hole_size)
    
    save_suffix = '.png'
    item = item + save_suffix
    if save:
        t1 = Time()
        filter_path = os.path.join(filter_dir, item)
        # pil_img = np_to_pil_crop(filtered_np_img)
        pil_img = np_to_pil(filtered_np_img)
        pil_img.save(filter_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Filtered", str(t1.elapsed()), filter_path))

        t1 = Time()
        mask_path = os.path.join(filter_mask_dir, item)
        pil_mask = np_to_pil(filtered_mask)
        pil_mask.save(mask_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Mask", str(t1.elapsed()), mask_path))

        t1 = Time()
        filter_cmp_path = os.path.join(filter_cmp_dir, item)
        pil_img_cmp = np_to_pil(filter_np_img_cmp)
        pil_img_cmp.save(filter_cmp_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Filtered CMP", str(t1.elapsed()), filter_cmp_path))

        t1 = Time()
        mask_cmp_path = os.path.join(filter_mask_cmp_dir, item)
        pil_mask_cmp = np_to_pil(filtered_mask_cmp)
        pil_mask_cmp.save(mask_cmp_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Mask CMP", str(t1.elapsed()), mask_cmp_path))

    percent = mask_percent(filtered_mask)
    print("Mask percent: {}%".format(percent))

    return percent


def apply_color_filter_to_dir(src, dst, save=True, display=False, hole_size=1000):
    t = Time()
    print("Applying filters to images\n")

    image_list = sorted(os.listdir(src))
    print('Number of images :  {}'.format(len(image_list)))
    # image_list = ['_20190403091259.png']
    # image_list = [image_list[11], image_list[22]]
    # image_list = ['_20190718215800.svs-080-32x-28662x73872-895x2308.png']

    FILTER_ING_DIR = src + '/filtering'
    FILTER_DIR = dst + '/filtered_img'
    FILTER_MASK_DIR = dst + '/filtered_mask'
    FILTER_CMP_DIR = dst + '/filtered_cmp_img'
    FILTER_MASK_CMP_DIR = dst + '/filtered_cmp_mask'

    if save:
        if not os.path.isdir(FILTER_ING_DIR):
            os.makedirs(FILTER_ING_DIR)
        if not os.path.isdir(FILTER_DIR):
            os.makedirs(FILTER_DIR)
        if not os.path.isdir(FILTER_MASK_DIR):
            os.makedirs(FILTER_MASK_DIR)
        if not os.path.isdir(FILTER_CMP_DIR):
            os.makedirs(FILTER_CMP_DIR)
        if not os.path.isdir(FILTER_MASK_CMP_DIR):
            os.makedirs(FILTER_MASK_CMP_DIR)

    percent = 0
    for item in tqdm(image_list):
        item_percent = apply_color_filter_to_image(item, src, FILTER_ING_DIR, FILTER_DIR, FILTER_MASK_DIR, FILTER_CMP_DIR, FILTER_MASK_CMP_DIR, 
                                                    save=True, display=False, hole_size=hole_size)
        percent += item_percent
    
    percent /= len(image_list)
    print("Average area of tissue over wsi is: {}%".format(percent))
    print("Time to apply filters to all images: %s\n" % str(t.elapsed()))


def apply_color_filter_to_image_v2(item, src, fitler_ing_dir,  filter_dir, filter_mask_dir, filter_cmp_dir, filter_mask_cmp_dir, 
                                save=True, display=False, hole_size=1000, object_size=200):
    t = Time()
    print('Processing slide:  {}'.format(item))
    
    image_path = os.path.join(src, item)
    # np_orig = open_pad_image_np(image_path)
    np_orig = open_image_np(image_path)
    item = item.split('.')[0]
    filtered_np_img, filtered_mask, filter_np_img_cmp, filtered_mask_cmp = color_filter_v2(np_orig, item, save_dir=fitler_ing_dir, save=False, display=display, hole_size=hole_size, object_size=object_size)
    
    save_suffix = '.png'
    item = item + save_suffix
    if save:
        t1 = Time()
        filter_path = os.path.join(filter_dir, item)
        # pil_img = np_to_pil_crop(filtered_np_img)
        pil_img = np_to_pil(filtered_np_img)
        pil_img.save(filter_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Filtered", str(t1.elapsed()), filter_path))

        t1 = Time()
        mask_path = os.path.join(filter_mask_dir, item)
        pil_mask = np_to_pil(filtered_mask)
        pil_mask.save(mask_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Mask", str(t1.elapsed()), mask_path))

        t1 = Time()
        filter_cmp_path = os.path.join(filter_cmp_dir, item)
        pil_img_cmp = np_to_pil(filter_np_img_cmp)
        pil_img_cmp.save(filter_cmp_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Filtered CMP", str(t1.elapsed()), filter_cmp_path))

        t1 = Time()
        mask_cmp_path = os.path.join(filter_mask_cmp_dir, item)
        pil_mask_cmp = np_to_pil(filtered_mask_cmp)
        pil_mask_cmp.save(mask_cmp_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Mask CMP", str(t1.elapsed()), mask_cmp_path))

    percent = mask_percent(filtered_mask)
    print("Mask percent: {}%".format(percent))

    return percent


def apply_color_filter_to_dir_v2(src, dst, save=True, display=False, hole_size=1000, object_size=200):
    t = Time()
    print("Applying filters to images\n")

    # image_list = sorted(os.listdir(src))
    image_list = ['2018-13135.png', '2018-14773.png']
    print('Number of images :  {}'.format(len(image_list)))
    # image_list = ['_20190403091259.png']
    # image_list = [image_list[11], image_list[22]]
    # image_list = ['_20190718215800.svs-080-32x-28662x73872-895x2308.png']

    FILTER_ING_DIR = dst + '/filtering'
    FILTER_DIR = dst + '/filtered_img'
    FILTER_MASK_DIR = dst + '/filtered_mask'
    FILTER_CMP_DIR = dst + '/filtered_cmp_img'
    FILTER_MASK_CMP_DIR = dst + '/filtered_cmp_mask'

    if save:
        if not os.path.isdir(FILTER_ING_DIR):
            os.makedirs(FILTER_ING_DIR)
        if not os.path.isdir(FILTER_DIR):
            os.makedirs(FILTER_DIR)
        if not os.path.isdir(FILTER_MASK_DIR):
            os.makedirs(FILTER_MASK_DIR)
        if not os.path.isdir(FILTER_CMP_DIR):
            os.makedirs(FILTER_CMP_DIR)
        if not os.path.isdir(FILTER_MASK_CMP_DIR):
            os.makedirs(FILTER_MASK_CMP_DIR)

    percent = 0
    for item in tqdm(image_list):
        item_percent = apply_color_filter_to_image_v2(item, src, FILTER_ING_DIR, FILTER_DIR, FILTER_MASK_DIR, FILTER_CMP_DIR, FILTER_MASK_CMP_DIR, 
                                                    save=True, display=False, hole_size=hole_size, object_size=object_size)
        percent += item_percent
    
    percent /= len(image_list)
    print("Average area of tissue over wsi is: {}%".format(percent))
    print("Time to apply filters to all images: %s\n" % str(t.elapsed()))

if __name__ == '__main__':
    root = '/media/ldy/7E1CA94545711AE6/OSCC-王/fine/processed data/tumor/'
    src_dir = root + '5x_png'
    dst_dir = root + '5x_filter'
    hole_size = 4000
    object_size = 15000
    apply_color_filter_to_dir_v2(src_dir, dst_dir, hole_size=hole_size, object_size=object_size)