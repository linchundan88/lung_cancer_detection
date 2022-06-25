'''
gen_mask: generating mask files based on ASAP annotation xml files.
'''

import sys
sys.path.append('/opt/ASAP/bin')  #pycharm neglect PYTHONPATH :(
import os
import numpy as np
import openslide
import tifffile
from skimage.transform import resize
import multiresolutionimageinterface as mir

RGB_min = (40, 40, 40)
RGB_max = (235, 210, 235)
colour_difference = 30



def gen_mask(file_wsi, file_xml, file_mask):
    print(f'generating {file_mask}...')
    os.makedirs(os.path.dirname(file_mask), exist_ok=True)

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(file_wsi)

    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(file_xml)
    xml_repository.load()

    annotation = annotation_list.getAnnotation(0)
    print(annotation.getArea())
    print(annotation.getNumberOfPoints())
    print(annotation.getCoordinate(0).getX())

    annotation_mask = mir.AnnotationToMask()
    # label_map = {'metastases': 1, 'normal': 2}  conversion_order = ['metastases', 'normal']
    annotation_mask.convert(annotation_list, file_mask, mr_image.getDimensions(), mr_image.getSpacing())

    # print(f'converting {file_mask} from 0-1 to 255...')
    # img1 = tifffile.imread(file_mask)
    # img1 = img1 * 255
    # tifffile.imwrite(file_mask, img1, compress=9)

    print(f'writing {file_mask} complete.')


def gen_tissue_mask(file_wsi, level, file_mask):

    slide = openslide.OpenSlide(file_wsi)
    # (width, height) = slide.level_dimensions[level]
    img_RGB = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'), dtype=np.int16)

    tissue_condition_max = (img_RGB[:, :, 0] > RGB_max[0]) & (img_RGB[:, :, 1] > RGB_max[1]) & (
                img_RGB[:, :, 2] > RGB_max[2])
    tissue_condition_max = ~tissue_condition_max

    tissue_condition_min = (img_RGB[:, :, 0] < RGB_min[0]) & (img_RGB[:, :, 1] < RGB_min[1]) & (
                img_RGB[:, :, 2] < RGB_min[2])
    tissue_condition_min = ~tissue_condition_min

    tissue_condition_color_diff = (np.abs(img_RGB[:, :, 0] - img_RGB[:, :, 1]) < colour_difference) \
                                  & (np.abs(img_RGB[:, :, 1] - img_RGB[:, :, 2]) < colour_difference) \
                                  & (np.abs(img_RGB[:, :, 0] - img_RGB[:, :, 2]) < colour_difference)
    tissue_condition_color_diff = ~tissue_condition_color_diff

    tissue_mask = tissue_condition_max & tissue_condition_min & tissue_condition_color_diff

    img_mask = np.array(tissue_mask).astype(np.uint8)
    img_mask *= 255

    print(f'writing {file_mask}...')
    os.makedirs(os.path.dirname(file_mask), exist_ok=True)
    tifffile.imwrite(file_mask, img_mask, compress=9)
    print(f'writing {file_mask} complete.')


def gen_tissue_mask_whole(file_wsi, level, file_mask):

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(file_wsi)
    level_dim = mr_image.getLevelDimensions(level)  #(width, height)
    level_ds = mr_image.getLevelDownsample(level)

    out_dims = mr_image.getLevelDimensions(0)
    step_size = int(512. /int(level_ds))
    writer = mir.MultiResolutionImageWriter()
    os.makedirs(os.path.dirname(file_mask), exist_ok=True)
    writer.openFile(file_mask)
    writer.setCompression(mir.LZW)
    writer.setDataType(mir.UChar)
    writer.setInterpolation(mir.NearestNeighbor)
    writer.setColorType(mir.Monochrome)
    writer.writeImageInformation(out_dims[0], out_dims[1])

    for y in range(0, level_dim[1], step_size):
        for x in range(0, level_dim[0], step_size):
            # getUCharPatch(x,y), however result:tile.shape (height, width)
            tile = mr_image.getUCharPatch(x, y, step_size, step_size, level)

            tissue_condition_max = (tile[:, :, 0] > RGB_max[0]) & (tile[:, :, 1] > RGB_max[1]) & (
                    tile[:, :, 2] > RGB_max[2])
            tissue_condition_max = ~tissue_condition_max

            tissue_condition_min = (tile[:, :, 0] < RGB_min[0]) & (tile[:, :, 1] < RGB_min[1]) & (
                    tile[:, :, 2] < RGB_min[2])
            tissue_condition_min = ~tissue_condition_min

            tissue_condition_color_diff = (np.abs(tile[:, :, 0] - tile[:, :, 1]) < colour_difference) \
                                          & (np.abs(tile[:, :, 1] - tile[:, :, 2]) < colour_difference) \
                                          & (np.abs(tile[:, :, 0] - tile[:, :, 2]) < colour_difference)
            tissue_condition_color_diff = ~tissue_condition_color_diff

            tissue_mask = tissue_condition_max & tissue_condition_min & tissue_condition_color_diff

            img_mask = np.array(tissue_mask).astype(np.uint8)
            img_mask *= 255

            write_tl = np.zeros((step_size, step_size), dtype='ubyte')
            write_tl[0:img_mask.shape[0], 0:img_mask.shape[1]] = img_mask

            res_tl = resize(write_tl, (512, 512), order=0, mode='constant', preserve_range=True).astype('ubyte')
            writer.writeBaseImagePart(res_tl.flatten())

    writer.finishImage()

