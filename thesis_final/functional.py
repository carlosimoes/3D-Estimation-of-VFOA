import torch
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import collections
import cv2

_cv2_pad_to_str = {'constant': cv2.BORDER_CONSTANT,
                   'edge': cv2.BORDER_REPLICATE,
                   'reflect': cv2.BORDER_REFLECT_101,
                   'symmetric': cv2.BORDER_REFLECT
                   }
_cv2_interpolation_to_str = {'nearest': cv2.INTER_NEAREST,
                             'bilinear': cv2.INTER_LINEAR,
                             'area': cv2.INTER_AREA,
                             'bicubic': cv2.INTER_CUBIC,
                             'lanczos': cv2.INTER_LANCZOS4}
_cv2_interpolation_from_str = {v: k for k, v in _cv2_interpolation_to_str.items()}

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not (_is_numpy_image(pic)):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    # handle numpy array
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor) or img.dtype == torch.uint8:
        return img.float().div(255)
    else:
        return img

def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    # This is faster than using broadcasting, don't change without benchmarking
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def resize(img, size, interpolation=cv2.INTER_LINEAR):
    r"""Resize the input numpy ndarray to the given size.
    Args:
        img (numpy ndarray): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    w, h, = size

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
    else:
        output = cv2.resize(img, dsize=size[::-1], interpolation=interpolation)
    if img.shape[2] == 1:
        return output[:, :, np.newaxis]
    else:
        return output
