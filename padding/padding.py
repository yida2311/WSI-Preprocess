import os 
from PIL import Image
from tqdm import tqdm

def imagePadding(img):
    w, h = img.size
    img_padded = Image.new('RGB', (w+10, h+10), color=(255, 255, 255))
    img_padded.paste(img, (5, 5, w+5, h+5))

    return img_padded 


if __name__ == '__main__':
    src = '/media/ldy/7E1CA94545711AE6/OSCC/5x_png'
    dst = '/media/ldy/7E1CA94545711AE6/OSCC/5x_padded5_png'
    for slide in tqdm(os.listdir(src)):
        img = Image.open(os.path.join(src, slide))
        img_padded = imagePadding(img)
        img_padded.save(os.path.join(dst, slide))