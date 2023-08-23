from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DEISMultistepScheduler,
    DDPMScheduler,
    DPMSolverSDEScheduler,
)
#用法
#scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP["ddim"])
#scheduler = scheduler_class.from_config(scheduler_config)
#Multistep相比single step来说，后者是不需要迭代的
SCHEDULER_MAP = dict(
    ddim=(DDIMScheduler, dict()),
    ddpm=(DDPMScheduler, dict()),
    deis=(DEISMultistepScheduler, dict()),
    lms=(LMSDiscreteScheduler, dict(use_karras_sigmas=False)),
    lms_k=(LMSDiscreteScheduler, dict(use_karras_sigmas=True)),
    pndm=(PNDMScheduler, dict()),
    heun=(HeunDiscreteScheduler, dict(use_karras_sigmas=False)),
    heun_k=(HeunDiscreteScheduler, dict(use_karras_sigmas=True)),
    euler=(EulerDiscreteScheduler, dict(use_karras_sigmas=False)),
    euler_k=(EulerDiscreteScheduler, dict(use_karras_sigmas=True)),
    euler_a=(EulerAncestralDiscreteScheduler, dict()),
    kdpm_2=(KDPM2DiscreteScheduler, dict()),
    kdpm_2_a=(KDPM2AncestralDiscreteScheduler, dict()),
    dpmpp_2s=(DPMSolverSinglestepScheduler, dict(use_karras_sigmas=False)),
    dpmpp_2s_k=(DPMSolverSinglestepScheduler, dict(use_karras_sigmas=True)),
    dpmpp_2m=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=False)),
    dpmpp_2m_k=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=True)),
    dpmpp_2m_sde=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=False, algorithm_type="sde-dpmsolver++")),
    dpmpp_2m_sde_k=(DPMSolverMultistepScheduler, dict(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")),
    dpmpp_sde=(DPMSolverSDEScheduler, dict(use_karras_sigmas=False, noise_sampler_seed=0)),
    dpmpp_sde_k=(DPMSolverSDEScheduler, dict(use_karras_sigmas=True, noise_sampler_seed=0)),
    unipc=(UniPCMultistepScheduler, dict(cpu_only=True)),
)

def getScheduler(name:str):
    """
    通过name获取采样器
    """
    scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get(name, SCHEDULER_MAP["ddim"])
    return scheduler_class.from_config(scheduler_extra_config)
    
