import cv2
import numpy as np
from PIL import Image
import torch

# 在diffuser中，白色表示重绘，黑色表示保留

#在外部可以这么导入
#import sys
#sys.path.append('/path/to/your/module_directory')

#蒙版反转
#import PIL.ImageOps
#mask = PIL.ImageOps.invert(mask)

#下载图片
#from diffusers.utils import load_image
#raw_image = load_image(
#   "image_path"
#)

def getCannyImage(image, low_threshold: int = 100, high_threshold: int = 200):
    """
    获取canny图片，用于传入controlnet的canny模型
    threshold取100到200是一个中间值，兼容大部分情况

    Args:
        image (PIL.Image): 原图
        low_threshold (int): 低阈值
        high_threshold (int): 高阈值

    Returns:
        PIL.Image: canny图片
    """

    # 
    # 将图像转为线条
    canny_image = cv2.Canny(np.array(image), low_threshold, high_threshold)
    # 但是返回的图片是单通道的，并且是w*h，我们需要变成3维
    canny_image = canny_image[:, :, None]
    # 同时转为RGB
    canny_image = np.concatenate(
        [canny_image, canny_image, canny_image], axis=2)
    # 接着再转为Image
    canny_image = Image.fromarray(canny_image)


def make_inpaint_condition(image, image_mask):
    """
    使用原图和遮罩图，合成用于inpaint的图

    """
    image=np.array(image).astype(np.float32) / 255.0
    image_mask=np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    #判断遮罩层中所有的元素，大于0.5为true，小于0.5为false
    #0为黑，255为白
    #这里将白色的像素区域设置为-1
    #https://github.com/huggingface/diffusers/pull/4207
    #白色表示重绘，黑色表示保留
    #print(image_mask)
    #print(image_mask > 0.5)
    image[image_mask > 0.5] = -1.0  # set as masked pixel

    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image
