# 测试图生图 + controlnet
# 实例化pipeline
# stablediffusionapi/majicmixrealistic

# 实例化的话，可以使用最通用的DiffusionPipeline
# 但是为了区分更具体的任务，我们可以使用StableDiffusionImg2ImgPipeline
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline,DPMSolverMultistepScheduler,ControlNetModel
from controlnet_aux.processor import Processor
from schedulers import getScheduler
from preProcesser import makeInpaintCondition
import matplotlib.pyplot as plt

# 这里可以直接使用hub上面有的，它会自己下载。
# 如果使用本地的model的话，则指定为本地的路径即可
repo_id = "stablediffusionapi/majicmixrealistic"

# 加载controlnet model，这里加载的是canny模型


# 采样器为DPM++ SDE karras
# safety_checker不为空的话，会很容易检测到NFSW，然后直接拒绝返回
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    repo_id,
    safety_checker = None,
    controlnet=[
      ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",torch_dtype=torch.float16),
      ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",torch_dtype=torch.float16)
    ],
    # 减少内存占用
    torch_dtype=torch.float16,
)

# controlnet pipeline
scheduler = getScheduler("dpmpp_sde_k")
pipe.scheduler = scheduler
# 这个会利用系统内存来帮忙预加载，但是会占用很多内存，在colab中会因为占用内存过高而炸了，这里关掉
#pipe.enable_model_cpu_offload()
#开启xformers提高速度，降低内存
pipe.enable_xformers_memory_efficient_attention()

#使用GPU计算
pipe.to("cuda")


pipe2 = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    repo_id,
    safety_checker = None,
    controlnet=[
      ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",torch_dtype=torch.float16),
      ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",torch_dtype=torch.float16),
      ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose",torch_dtype=torch.float16),
    ],
    # 减少内存占用
    torch_dtype=torch.float16,
)
pipe2.scheduler = scheduler
# 这个会利用系统内存来帮忙预加载，但是会占用很多内存，在colab中会因为占用内存过高而炸了，这里关掉
#pipe.enable_model_cpu_offload()
#开启xformers提高速度，降低内存
pipe2.enable_xformers_memory_efficient_attention()

#使用GPU计算
pipe2.to("cuda")

# 预处理器
cannyProcessor = Processor('canny')
#lineartRealisticProcessor = Processor('lineart_realistic')

import PIL
import requests
import torch
from io import BytesIO
# 下载图片
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

from PIL import Image
import PIL.ImageOps
#image = Image.open("image.jpg")
#maskImage2 = Image.open("maskImage.png")
openpose_image = Image.open("dwPose2.png")
#maskImage2=PIL.ImageOps.invert(maskImage2.convert("RGB"))
#plt.imshow(maskImage2)
#plt.axis('off')  # 关闭坐标轴
#plt.show()
#import gc
#del pipe
#del pipe2
#gc.collect()
#torch.cuda.empty_cache()


from diffusers.utils import load_image,make_image_grid
import PIL.ImageOps
import random


def img2imgInpaint(prompt:str,
                   negative_prompt:str,
                   generator,mypipe,control_image,
                   image,mask_image,step=20,guidance_scale=7.5,strength=0.8,width=512,height=512,
                   ):
    canny_image=cannyProcessor(image)
    inpaint_image=makeInpaintCondition(image,mask_image)


    return mypipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=step,
        guidance_scale=guidance_scale,
        strength=strength,
        width=width,
        height=height,
        generator=torch.manual_seed(0),
        control_image=[canny_image,inpaint_image]
    ).images[0]


def xyz():
    # 对比：
    # inpaint+canny
    # inpaint
    # 进行xyz的时候，种子需要固定，对比3次
    

    prompt=''
    negativePrompt=''

    width=600
    height=800
    image=''
    maskImage=''
    
    image=download_image(image)
    maskImage=download_image(maskImage)
    maskImage = PIL.ImageOps.invert(maskImage)

    canny_image=cannyProcessor(image)
    inpaint_image=makeInpaintCondition(image,maskImage)
    #lineart_image=lineartRealisticProcessor(image)
    
    imgs=[]
    imgs2=[]
    for i in range(3):
      rand=random.randint(1, 10000)
      generator=torch.manual_seed(rand)
      img1=pipe(
        prompt=prompt,
        negative_prompt=negativePrompt,
        image=image,
        mask_image=maskImage,
        num_inference_steps=20,
        guidance_scale=7.5,
        strength=0.8,
        width=width,
        height=height,
        generator=generator,
        control_image=[canny_image,inpaint_image]
      ).images[0]
      generator=torch.manual_seed(rand)
      img2=pipe2(
        prompt=prompt,
        negative_prompt=negativePrompt,
        image=image,
        mask_image=maskImage,
        num_inference_steps=20,
        guidance_scale=7.5,
        strength=0.8,
        width=width,
        height=height,
        generator=generator,
        control_image=[canny_image,inpaint_image,openpose_image]
      ).images[0]
      imgs.append(img1)
      imgs2.append(img2)
    
    img3=make_image_grid(imgs+imgs2,2,len(imgs))
    img3.save('img3.png')
    plt.imshow(img3)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
xyz()