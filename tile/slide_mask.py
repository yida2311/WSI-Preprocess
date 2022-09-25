import os
import math
from glob import glob
import cv2
import json
import openslide
from tqdm import tqdm
from PIL import Image
import numpy as np
from xml.dom import minidom


def generate_mask(slide, tissue_mask_dir, anno_mask_dir, std_mask_dir, rgb_mask_dir):
    """Generate std mask & rgb mask for segmentation and visualization"""
    tissue_mask_path = os.path.join(tissue_mask_dir, slide+'.png')
    anno_mask_path = os.path.join(anno_mask_dir, slide+'.png')
    
    tissue_mask = np.array(Image.open(tissue_mask_path).convert('1'))
    anno_mask = np.array(Image.open(anno_mask_path))
    std_mask = tissue_mask * anno_mask + tissue_mask
    rgb_mask = class_to_RGB(std_mask)

    std_mask = Image.fromarray(std_mask)
    rgb_mask = Image.fromarray(rgb_mask)
    std_mask.save(os.path.join(std_mask_dir, slide+'.png'))
    rgb_mask.save(os.path.join(rgb_mask_dir, slide+'.png'))


def generate_anno_mask(slide, scale, img_dir, xml_dir, save_dir, img_type='.png'):
    """generate annotation mask """
    img_path = os.path.join(img_dir, slide+img_type)
    xml_path = os.path.join(xml_dir, slide+ '.xml')

    if img_type == '.svs':
        img = openslide.OpenSlide(img_path)
        w, h = img.dimensions
        w, h = math.floor(w/scale), math.floor(h/scale)
    elif img_type == '.png' or '.tif':
        img = cv2.imread(img_path)
        h, w, _ = img.shape
    else:
        raise ValueError

    print(h, w)
    mask = np.zeros((h, w), dtype=np.uint8)

    regions, labels = get_regions(xml_path, scale=scale)
    for region, label in zip(regions, labels):
        region = region.reshape(-1, 1, 2)
        cv2.polylines(mask, np.int32([region]), True, label)
        cv2.fillPoly(mask, np.int32([region]), label)
    
    cv2.imwrite(os.path.join(save_dir, slide+'.png'), mask)


def get_regions(xml_path, scale=4):
    ''' Parses the xml at the given path, assuming annotation format importable by ImageScope. '''
    xml = minidom.parse(xml_path)
    region_coord, region_labels = [], []
    # The first region marked is always the tumour delineation
    annotations = xml.getElementsByTagName('Annotation')
    for annotation in annotations:
        regions = annotation.getElementsByTagName('Region')
        if regions is None or len(regions) == 0:
            continue
        elif regions[0].getAttribute('Text') == "":
            continue
        else:
            for region in regions:
                vertices = region.getElementsByTagName('Vertex')
                attribute = region.getElementsByTagName("Attribute")
                if len(attribute) > 0:
                    r_label = attribute[0].attributes['Value'].value
                else:
                    r_label = int(region.getAttribute('Text')) + 1
                region_labels.append(r_label)

                # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
                coords = np.zeros((len(vertices), 2))
                for i, vertex in enumerate(vertices):
                    coords[i][0] = int(vertex.attributes['X'].value) // scale 
                    coords[i][1] = int(vertex.attributes['Y'].value) // scale 
                region_coord.append(coords)
            
    return region_coord, region_labels


def class_to_RGB(label):
    h, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(h, w, 3)).astype(np.uint8)

    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 0]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]

    return colmap


if __name__ == '__main__':
    IMG_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/5x_cut/'
    XML_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/svs/svs_40x/'
    ANNO_MASK_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/temp/anno_mask'
    STD_MASK_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/temp/std_mask/'
    RGB_MASK_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/temp/rgb_mask/'
    TISSUE_MASK_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/5x_filter/filtered_mask/' 
    SCALE = 8

    # slide_list = os.listdir(TISSUE_MASK_DIR)
    # slide_list = [c.split('.')[0] for c in slide_list]
    # slide_list = glob(XML_DIR+'*.xml')
    # slide_list = sorted([c.split('/')[-1].split('.')[0] for c in slide_list])
    # slide_list = ['2018-16294']
    # slide_list = ['2020-07912', '2020-07912-2']  # ', 
    slide_list = ['2017-05588']

    for slide in tqdm(slide_list):
        print(slide)
        generate_anno_mask(slide, SCALE, IMG_DIR, XML_DIR, ANNO_MASK_DIR)
    
    for slide in slide_list:
        generate_mask(slide, TISSUE_MASK_DIR, ANNO_MASK_DIR, STD_MASK_DIR, RGB_MASK_DIR)
    
