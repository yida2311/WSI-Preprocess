import os 
from tqdm import tqdm 
from PIL import Image
import math

def get_patch_info(shape, p_size, overlap):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    step_x = step_y = p_size - overlap
    n = math.ceil((x-overlap) / step_x)
    m = math.ceil((y-overlap) / step_y)
    
    step_x = (x - p_size) * 1.0 / (n - 1) if n > 1 else step_x
    step_y = (y - p_size) * 1.0 / (m - 1) if m > 1 else step_y
#     print(n, n, step_x, step_y)
    return n, m, step_x, step_y         


def get_local_patches(pil_img, p_size, overlap):
    patches = []
    coords = [] # h, w
    w, h = pil_img.size
    col, ver, step_col, step_ver = get_patch_info([w, h], p_size, overlap)
    for i in range(ver):
        if i < ver-1:
            coord_ver = int(i * step_ver)
        else:
            coord_ver = int(h - p_size)
        for j in range(col):
            if j < col - 1:
                coord_col = int(j * step_col)
            else:
                coord_col = int(w - p_size)
            
            patch = pil_img.crop((coord_col, coord_ver, coord_col+p_size, coord_ver+p_size))
            patches.append(patch)
            coords.append([i, j])
    
    return patches, coords

def get_global_patch(pil_img, p_size):
    patch = pil_img.resize((p_size, p_size))
    return patch   
             

root = '/disk2/ldy/'
terms = ['patch','std_mask', 'rgb_mask']
p_size = 800
overlap = 60

for term in terms:
    src = root + '5x_3020-v2/' + term
    dst = root + '5x_gl_3020-v2/' + term
    if not os.path.exists(dst):
        os.makedirs(dst)

    for slide in tqdm(sorted(os.listdir(src))):
        src_slide_dir = os.path.join(src, slide)
        dst_slide_dir = os.path.join(dst, slide)
        if not os.path.exists(dst_slide_dir):
            os.makedirs(dst_slide_dir)
            
        for tile in os.listdir(src_slide_dir):
            tile_img = Image.open(os.path.join(src_slide_dir, tile))
            g_patch = get_global_patch(tile_img, p_size)
            l_patches, coords = get_local_patches(tile_img, p_size, overlap)
            
            g_patch.save(os.path.join(dst_slide_dir, tile))
            tile_term = tile.split('.')[0][:-1]
            dst_patch_dir = os.path.join(dst_slide_dir, tile_term)
            if not os.path.exists(dst_patch_dir):
                os.makedirs(dst_patch_dir)
            for l_patch, coord in zip(l_patches, coords):
                patch_name = tile_term + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + '.png'
                l_patch.save(os.path.join(dst_patch_dir, patch_name))