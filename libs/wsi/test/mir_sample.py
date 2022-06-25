import sys
sys.path.append('/opt/ASAP/bin')  #pycharm neglect PYTHONPATH :(
import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.filters import median_filter
from skimage.transform import resize

def read_wsi():
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open('/devdata_b/data/ACDC_2019/Training1/Images/1.tif')
    level = 2
    ds = mr_image.getLevelDownsample(level)
    image_patch = mr_image.getUCharPatch(int(568 * ds), int(732 * ds), 300, 200, level)

    dims = mr_image.getLevelDimensions(6)  #(width, height)

    tile = mr_image.getUCharPatch(0, 0, dims[0], dims[1], 6)
    cv2.imwrite('/tmp/aaa.jpg', tile)
    plt.imshow(tile)

def read_wsi_xml():
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open('/devdata_b/data/ACDC_2019/Training1/Images/1.tif')

    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource('/devdata_b/data/ACDC_2019/Training1/Annotation/1.xml')
    xml_repository.load()

    annotation = annotation_list.getAnnotation(0)
    print(annotation.getArea())
    print(annotation.getNumberOfPoints())
    print(annotation.getCoordinate(0).getX())

    annotation_mask = mir.AnnotationToMask()
    # label_map = {'metastases': 1, 'normal': 2}
    output_path = '/devdata_b/data/ACDC_2019/Training1/Images/1_output.tif'
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing())


def write_tiff():
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open('/devdata_b/data/ACDC_2019/Training1/Images/1.tif')
    level = 3
    level_dim = mr_image.getLevelDimensions(level)
    level_ds = mr_image.getLevelDownsample(level)
    tile = mr_image.getUCharPatch(0, 0, level_dim[0], level_dim[1], level)

    tile_clipped = np.clip(tile, 1, 254)
    tile_od = -np.log(tile_clipped / 255.)
    D = median_filter(np.sum(tile_od, axis=2) / 3., size=3)
    raw_mask = (((D > 0.02 * -np.log(1/255.)) *
                (D < 0.98 * -np.log(1 / 255.))).astype("ubyte"))

    out_dims = mr_image.getLevelDimensions(0)
    step_size = int(512. /int(level_ds))
    writer = mir.MultiResolutionImageWriter()
    writer.openFile('/devdata_b/data/ACDC_2019/Training1/Images/1_mask.tif')
    writer.setCompression(mir.LZW)
    writer.setDataType(mir.UChar)
    writer.setInterpolation(mir.NearestNeighbor)
    writer.setColorType(mir.Monochrome)
    writer.writeImageInformation(out_dims[0], out_dims[1])

    for y in range(0, level_dim[1], step_size):
        for x in range(0, level_dim[0], step_size):
            write_tl = np.zeros((step_size, step_size), dtype='ubyte')
            cur_tl = raw_mask[y:y+step_size, x:x+step_size]
            write_tl[0:cur_tl.shape[0], 0:cur_tl.shape[1]] = cur_tl

            res_tl = resize(write_tl, (512, 512), order=0, mode='constant', preserve_range=True).astype('ubyte')
            writer.writeBaseImagePart(res_tl.flatten())
    writer.finishImage()


write_tiff()


print('OK')

