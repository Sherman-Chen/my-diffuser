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

#https://github.com/patrickvonplaten/controlnet_aux
#https://github.com/invoke-ai/InvokeAI/blob/0909812c84570d7998bfd90ebd136f252650a909/invokeai/app/invocations/controlnet_image_processors.py#L232
#不需要自己封装，直接使用上面的即可，已经有人封装好了
#用法
"""
from PIL import Image
import requests
from io import BytesIO
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load checkpoints
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
mobile_sam = SamDetector.from_pretrained("dhkim2810/MobileSAM", model_type="vit_t", filename="mobile_sam.pt")
leres = LeresDetector.from_pretrained("lllyasviel/Annotators")

# instantiate
canny = CannyDetector()
content = ContentShuffleDetector()
face_detector = MediapipeFaceDetector()


# process
processed_image_hed = hed(img)
processed_image_midas = midas(img)
processed_image_mlsd = mlsd(img)
processed_image_open_pose = open_pose(img, hand_and_face=True)
processed_image_pidi = pidi(img, safe=True)
processed_image_normal_bae = normal_bae(img)
processed_image_lineart = lineart(img, coarse=True)
processed_image_lineart_anime = lineart_anime(img)
processed_image_zoe = zoe(img)
processed_image_sam = sam(img)
processed_image_leres = leres(img)

processed_image_canny = canny(img)
processed_image_content = content(img)
processed_image_mediapipe_face = face_detector(img)

coarse：是否使用简易模式
"""

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


def makeInpaintCondition(image, image_mask):
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
