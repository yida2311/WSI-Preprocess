import os
import math 
import cv2
import glob
import numpy as np
from xml.dom import minidom
from tqdm import tqdm


def cut_image(slide, img_dir, xml_dir, save_dir, scale=4):
    img_path = os.path.join(img_dir, slide+'.png')
    xml_path = os.path.join(xml_dir, slide+'.xml')
    save_path = os.path.join(save_dir, slide+'.png')

    img = cv2.imread(img_path)
    regions = get_cut_regions(xml_path, scale=scale)
    if len(regions) > 0:
        h, w, c = img.shape
        mask = np.ones((h, w, c), dtype=np.uint8)
        for region in regions:
            region = region.reshape(-1, 1, 2)
            cv2.polylines(mask, np.int32([region]), True, 0)
            cv2.fillPoly(mask, np.int32([region]), 0)
        img = img * mask + 255*(1 - mask)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, img)


def get_cut_regions(xml_path, scale=4):
    ''' Parses the xml at the given path, assuming annotation format importable by ImageScope. '''
    xml = minidom.parse(xml_path)
    annotations = xml.getElementsByTagName('Annotation')
    region_coord = []
    for annotation in annotations:
        if annotation.getAttribute('Name') == 'cut':
            print('cut')
            regions = annotation.getElementsByTagName('Region')
            if regions is None or len(regions) == 0:
                continue
            for region in regions:
                vertices = region.getElementsByTagName('Vertex')
                # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
                coords = np.zeros((len(vertices), 2))
                for i, vertex in enumerate(vertices):
                    coords[i][0] = int(vertex.attributes['X'].value) // scale
                    coords[i][1] = int(vertex.attributes['Y'].value) // scale
                region_coord.append(coords)
    
    return region_coord



if __name__ == '__main__':
    IMG_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/5x_png/'
    XML_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/svs/svs_20x/'
    CUT_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/5x_cut/'
    SCALE = 4

    slide_list = sorted(glob.glob(XML_DIR+ '*.xml'))
    slide_list = [c.split('/')[-1].split('.')[0] for c in slide_list]

    for slide in tqdm(slide_list):
        print(slide)
        cut_image(slide, IMG_DIR, XML_DIR, CUT_DIR, scale=SCALE)

 



