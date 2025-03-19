import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import gryds
import scipy.ndimage as ndi
from scipy.stats import norm
import warnings


warnings.filterwarnings("ignore")


def aug_histogram_equalization(image1, seg, image2):
    img_min1, img_max1 = image1.min(), image1.max()
    img_min2, img_max2 = image2.min(), image2.max()

    image1_flattened = image1.flatten()
    hist1, bins1 = np.histogram(image1_flattened, bins=256, range=[image1_flattened.min(), image1_flattened.max()])
    cdf1 = hist1.cumsum()
    cdf_normalized1 = cdf1 * (hist1.max() / cdf1.max())
    image1 = np.interp(image1_flattened, bins1[:-1], cdf_normalized1).reshape(image1.shape)
    image1 = np.interp(image1, (image1.min(), image1.max()), (img_min1, img_max1))

    image2_flattened = image2.flatten()
    hist2, bins2 = np.histogram(image2_flattened, bins=256, range=[image2_flattened.min(), image2_flattened.max()])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * (hist2.max() / cdf2.max())
    image2 = np.interp(image2_flattened, bins2[:-1], cdf_normalized2).reshape(image2.shape)
    image2 = np.interp(image2, (image2.min(), image2.max()), (img_min2, img_max2))

    return image1, seg, image2

def aug_transform(image1, transform):
    img_min1, img_max1 = image1.min(), image1.max()


    image1 = (image1 - image1.mean()) / image1.std()
    image1 = np.interp(image1, (image1.min(), image1.max()), (0, 1))

    image1 = transform(image1)
    

    image1 = np.interp(image1, (image1.min(), image1.max()), (img_min1, img_max1))
    

    return image1

def aug_log(image1):
    return aug_transform(image1, lambda x: np.log(1 + x))

def aug_sqrt(image1 ):
    return aug_transform(image1,  np.sqrt)

def aug_sin(image1):
    return aug_transform(image1,  np.sin)

def aug_exp(image1):
    return aug_transform(image1, np.exp)

def aug_sig(image1):
    return aug_transform(image1, lambda x: 1 / (1 + np.exp(-x)))

def aug_laplace(image1):
    return aug_transform(image1,  lambda x: np.abs(ndi.laplace(x)))

def aug_inverse(image1):
    image1 = image1.min() + image1.max() - image1
    return image1

def aug_bspline(image1, seg, image2):
    grid = rs.rand(3, 3, 3, 3)
    bspline = gryds.BSplineTransformation((grid - .5) / 5)
    grid[:, 0] += ((grid[:, 0] > 0) * 2 - 1) * .9
    image1 = gryds.Interpolator(image1).transform(bspline).astype(np.float64)
    image2 = gryds.Interpolator(image2).transform(bspline).astype(np.float64)
    seg = gryds.Interpolator(seg, order=0).transform(bspline).astype(np.uint8)
    return image1, seg, image2

def aug_flip(image1, seg, image2):
    subject = tio.RandomFlip(axes=('LR',))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_aff(image1, seg, image2):
    subject = tio.RandomAffine()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_elastic(image1, seg, image2):
    subject = tio.RandomElasticDeformation(max_displacement=40)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_anisotropy(image1, seg, image2, downsampling=7):
    subject = tio.RandomAnisotropy(downsampling=downsampling)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_motion(image1, seg, image2):
    subject = tio.RandomMotion()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_ghosting(image1, seg, image2):
    subject = tio.RandomGhosting()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_spike(image1, seg, image2):
    subject = tio.RandomSpike(intensity=(1, 2))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_bias_field(image1, image2, seg):
    subject = tio.RandomBiasField()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_blur(image1, seg, image2):
    subject = tio.RandomBlur()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_noise(image1, seg, image2):
    original_mean1, original_std1 = np.mean(image1), np.std(image1)
    original_mean2, original_std2 = np.mean(image2), np.std(image2)

    image1 = (image1 - original_mean1) / original_std1
    image2 = (image2 - original_mean2) / original_std2

    subject = tio.RandomNoise()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    image1 = image1 * original_std1 + original_mean1
    image2 = image2 * original_std2 + original_mean2

    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_swap(image1, seg, image2):
    subject = tio.RandomSwap()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_labels2image(image1, seg, image2, leave_background=0.5, classes=None):
    _seg = seg
    if classes:
        _seg = combine_classes(seg, classes)
    subject = tio.RandomLabelsToImage(label_key="seg", image_key="image")(tio.Subject(
        seg=tio.LabelMap(tensor=np.expand_dims(_seg, axis=0))
    ))
    new_img = subject.image.data.squeeze().numpy().astype(np.float64)

    if rs.rand() < leave_background:
        img_min1, img_max1 = np.min(image1), np.max(image1)
        _image1 = (image1 - img_min1) / (img_max1 - img_min1)

        new_img_min, new_img_max = np.min(new_img), np.max(new_img)
        new_img = (new_img - new_img_min) / (new_img_max - new_img_min)
        new_img[_seg == 0] = _image1[_seg == 0]
        new_img = np.interp(new_img, (new_img.min(), new_img.max()), (img_min1, img_max1))

    return new_img, seg, image2

def aug_gamma(image1, seg, image2):
    subject = tio.RandomGamma()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)


def parse_class(c):
    c = [_.split('-') for _ in c.split(',')]
    c = tuple(__ for _ in c for __ in list(range(int(_[0]), int(_[-1]) + 1)))
    return c

def combine_classes(seg, classes):
    _seg = np.zeros_like(seg)
    for i, c in enumerate(classes):
        _seg[np.isin(seg, c)] = i + 1
    return _seg

if __name__ == '__main__':
    main()