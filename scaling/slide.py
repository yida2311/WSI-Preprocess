import os
import re
import sys
import glob
import math
import gc

import openslide
import PIL
import numpy as np 
from PIL import Image

from util import Time, _load_image_lessthan_2_29, _load_image_morethan_2_29


def func_read_region(h, w):
  if (h*w) >= 2**29:
        openslide.lowlevel._load_image = _load_image_morethan_2_29
  else:
        openslide.lowlevel._load_image = _load_image_lessthan_2_29


def open_slide(filename):
  """
  Open a whole-slide image (*.tif, etc).

  Args:
    filename: Name of the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide


def save_thumbnail(pil_img, size, path):
  """
  Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.

  Args:
    pil_img: The PIL image to save as a thumbnail.
    size:  The maximum width or height of the thumbnail.
    path: The path to the thumbnail.
  """
  max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
  img = pil_img.resize(max_size, PIL.Image.BILINEAR)
  img.save(path)


def slide_to_scaled_pil_image(slide_path, scale=16):
    """
    Convert a WSI training slide to a scaled-down PIL image.

    Args:
        slide_path

    Returns:
        Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    print("Opening Slide %s" % (slide_path))
    slide = open_slide(slide_path)

    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / scale)
    new_h = math.floor(large_h / scale)
    level = slide.get_best_level_for_downsample(scale)
    w_rec, h_rec = slide.level_dimensions[level]

    if w_rec*h_rec < 2**31:
        func_read_region(h_rec, w_rec)
        whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
        whole_slide_image = whole_slide_image.convert("RGB")
        img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        del whole_slide_image
        gc.collect()
    else:
        print('subimg')
        num_subimg = math.ceil(w_rec*h_rec / (2**30))
        h_rec_new = math.ceil(h_rec / num_subimg)
        new_h_divide = math.floor(large_h / scale/ num_subimg)
        # new_h = new_h_divide * num_subimg

        func_read_region(h_rec_new, w_rec)
        img = Image.new("RGB", (new_w, new_h))

        for i in range(num_subimg):
            print(i)
            if i < num_subimg:
                whole_slide_image = slide.read_region((0, i*h_rec_new), level, (w_rec, h_rec_new))
            else:
                whole_slide_image = slide.read_region((0, i*h_rec_new), level, (w_rec, h_rec - i*h_rec_new))
            whole_slide_image = whole_slide_image.convert("RGB")
            subimg = whole_slide_image.resize((new_w, new_h_divide), PIL.Image.BILINEAR)
            print(whole_slide_image.size)
            img.paste(subimg, (0, i*new_h_divide))

        del whole_slide_image
        gc.collect()

    print("reading done!")
    return img, large_w, large_h, new_w, new_h


def training_slide_to_image(slide_name, scale, slide_dir, img_dir, thm_dir, save_format='.png'):
    slide_path = os.path.join(slide_dir, slide_name+'.svs')
    img_path = os.path.join(img_dir, slide_name+save_format)
    thm_path = os.path.join(thm_dir, slide_name+'.png')
    # print(img_path)
    # exit(0)

    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_path, scale)
    print("Saving image to: " + img_path)
    
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(thm_dir):
        os.makedirs(thm_dir)
    img.save(img_path)
    save_thumbnail(img, 512, thm_path)


def training_slides_to_images(scale, slide_dir, img_dir, thm_dir, save_format='.png'):
    t = Time()
    names = glob.glob(slide_dir+'*.svs') 
    names = [c.split('/')[-1].split('.')[0] for c in names]
    print('Num of slides: {}'.format(len(names)))
    for slide_name in names:
        training_slide_to_image(slide_name, scale, slide_dir, img_dir, thm_dir, save_format)
    t.elapsed_display()


def save_thumbnail(pil_img, size, path):
  """
  Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.

  Args:
    pil_img: The PIL image to save as a thumbnail.
    size:  The maximum width or height of the thumbnail.
    path: The path to the thumbnail.
  """
  max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
  img = pil_img.resize(max_size, PIL.Image.BILINEAR)
  img.save(path)


if __name__ == '__main__':
    SCALE = 8
    FORMAT = '.png'
    SLIDE_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/orig_data/'
    IMG_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/scaledown8_png/'
    THM_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/thumbnail/'

    training_slides_to_images(SCALE, SLIDE_DIR, IMG_DIR, THM_DIR, save_format=FORMAT)


