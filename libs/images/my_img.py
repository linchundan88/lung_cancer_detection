'''
 image_in_thresholds: determine whether the mean of the input image is with image_thresholds
 image_pad: adding black padding for patche images at the border of the WSI, so that these patch images can match the input size  of neural networks.
 image_crop: cropping image to the pre-defined size.
 save_tiled_tiff: save whole slide image, ASAP and other software can open it.
'''
import gc
import os
import cv2
import numpy as np
import tifffile
from math import floor, ceil



def image_in_thresholds(img, image_thresholds):
    img_mean = np.mean(img)
    if img_mean < image_thresholds[0] or img_mean > image_thresholds[1]:
        return False
    else:
        return True


def image_pad(image1, height_output, width_output):

    if image1.ndim == 3:
        channel = image1.shape[2]
    elif image1.ndim == 2:
        image1 = np.expand_dims(image1, axis=-1)
        channel = 1
    else:
        raise ValueError('the number of channel is error!')


    if (image1.shape[0:2]) == (height_output, width_output):
        return image1
    else:
        height, width = image1.shape[0:2]

        img_mean = np.mean(image1)

        if height_output > height:
            padding_top = floor((height_output - height) / 2)
            padding_bottom = ceil((height_output - height) / 2)

            image_padding_top = np.ones((padding_top, width, channel), dtype=np.uint8)
            image_padding_top *= img_mean
            image_padding_bottom = np.ones((padding_bottom, width, channel), dtype=np.uint8)
            image_padding_bottom *= img_mean

            image1 = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)

            height, width = image1.shape[0:2]

        if width_output > width:
            padding_left = floor((width_output - width) / 2)
            padding_right = ceil((width_output - width) / 2)

            image_padding_left = np.ones((height, padding_left, channel), dtype=np.uint8)
            image_padding_left *= img_mean
            image_padding_right = np.ones((height, padding_right, channel), dtype=np.uint8)
            image_padding_right *= img_mean

            image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)

            # height, width = image1.shape[0:2]

        return image1


def image_crop(image1, height_output, width_output):
    height, width = image1.shape[:2]

    image1 = image1[0:min(height, height_output), 0:min(width, width_output)]

    return image1


def save_tiled_tiff(filename, image, tile_size=(256, 256), compress_ratio=None):
    print(f'writing image file {filename}')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # img = img.astype(np.uint8)

    h, w = image.shape[0:2]

    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        level = 0
        while True:
            print(f'generationg level: {level} of {filename}...')
            if compress_ratio is not None:
                tif.write(
                    image,
                    # software='Glencoe/Faas pyramid',
                    metadata=None,
                    tile=tile_size,
                    # resolution=(1000/2**level, 1000/2**level, 'CENTIMETER'),
                    compression=('JPEG', compress_ratio),  # requires imagecodecs #, compress_ratio
                    # subfiletype=1 if level else 0,
                )
            else:
                tif.write(
                    image,
                    # software='Glencoe/Faas pyramid',
                    metadata=None,
                    tile=tile_size,
                    # resolution=(1000/2**level, 1000/2**level, 'CENTIMETER'),
                    # subfiletype=1 if level else 0,
                )

            if max(w, h) < min(tile_size):
                break

            level += 1
            w //= 2
            h //= 2
            image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            gc.collect()


