import os
import cv2
import numpy as np

from util import Time, open_image, open_image_np, np_to_pil, mask_rgb, mask_percent, mask_percentage_text, display_img, findHoles
import basic_filter as filter


def color_filter(np_img, item, save_dir=None, save=False, display=False, hole_size=1000):
    rgb = np_img
    mask_not_green = filter.filter_green_channel(rgb)
    rgb_not_green = mask_rgb(rgb, mask_not_green)

    mask_not_gray = filter.filter_grays(rgb, tolerance=13)
    rgb_not_gray = mask_rgb(rgb, mask_not_gray)

    mask_not_red_pen = filter.filter_red_pen(rgb)
    rgb_not_red_pen = mask_rgb(rgb, mask_not_red_pen)

    mask_not_green_pen = filter.filter_green_pen(rgb)
    rgb_not_green_pen = mask_rgb(rgb, mask_not_green_pen)

    mask_not_blue_pen = filter.filter_blue_pen(rgb)
    rgb_not_blue_pen = mask_rgb(rgb, mask_not_blue_pen)

    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_not_red_pen & mask_not_green_pen & mask_not_blue_pen
    rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)
    
    mask_remove_holes = filter.filter_remove_small_holes(mask_not_gray , min_size=hole_size, output_type="bool")
    rgb_remove_holes = mask_rgb(rgb, mask_remove_holes)

    # mask_remove_obejcts = filter.filter_remove_small_objects(mask_remove_holes, min_size=object_size, output_type="bool")
    # rgb_remove_obejcts = mask_rgb(rgb, mask_remove_obejcts)
    
    n, label, stats, centrids = cv2.connectedComponentsWithStats(np.array(mask_remove_holes, dtype='uint8'), connectivity=8)
    index = np.argmax(stats[1:, 4])+1
    mask_remove_isolated = np.zeros_like(mask_remove_holes, dtype='bool')
    mask_remove_isolated[label==index] = 1
    # holes = findHoles(mask_remove_isolated)
    rgb_remove_isolated = mask_rgb(rgb, mask_remove_isolated)
    # rgb_remove_isolated = mask_rgb(rgb, mask_remove_isolated+holes)
    # mask_remove_isolated = mask_remove_isolated - holes

    return rgb_remove_isolated, mask_remove_isolated, rgb_remove_holes, mask_remove_holes


def color_filter_v2(np_img, item, save_dir=None, save=False, display=False, hole_size=1000, object_size=200):
    rgb = np_img
    mask_not_green = filter.filter_green_channel(rgb)
    rgb_not_green = mask_rgb(rgb, mask_not_green)

    mask_not_gray = filter.filter_grays(rgb, tolerance=13)
    rgb_not_gray = mask_rgb(rgb, mask_not_gray)

    mask_not_red_pen = filter.filter_red_pen(rgb)
    rgb_not_red_pen = mask_rgb(rgb, mask_not_red_pen)

    mask_not_green_pen = filter.filter_green_pen(rgb)
    rgb_not_green_pen = mask_rgb(rgb, mask_not_green_pen)

    mask_not_blue_pen = filter.filter_blue_pen(rgb)
    rgb_not_blue_pen = mask_rgb(rgb, mask_not_blue_pen)

    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_not_red_pen & mask_not_green_pen & mask_not_blue_pen
    rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)
    
    mask_remove_holes = filter.filter_remove_small_holes(mask_not_gray , min_size=hole_size, output_type="bool")
    rgb_remove_holes = mask_rgb(rgb, mask_remove_holes)

    mask_remove_obejcts = filter.filter_remove_small_objects(mask_remove_holes, min_size=object_size, output_type="bool")
    rgb_remove_obejcts = mask_rgb(rgb, mask_remove_obejcts)

    return rgb_remove_obejcts, mask_remove_obejcts, rgb_remove_holes, mask_remove_holes


def save_display(np_img, dir, item, save, display, filter_num, display_text, file_text, 
                display_mask_percentage=True):
    mask_percentage = None
    if display_mask_percentage:
        mask_percentage = mask_percent(np_img)
        display_text = display_text + '\n(' + mask_percentage_text(mask_percentage) + ' masked)'
    if filter_num is None:
        display_text = item + '_' + display_text
    else:
        display_text = item + '_' + str(filter_num) + '_' + display_text
    
    if display:
        display_img(np_img, display_text)
    
    if save:
        file_text = item + '_' + str(filter_num) + '_' + file_text + '.png'
        t = Time()
        file_path = os.path.join(dir, file_text)
        pil_img = np_to_pil(np_img)
        pil_img.save(file_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), file_path))


