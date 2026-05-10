import sys, os, warnings
warnings.filterwarnings("ignore")

APP_DIR   = r"C:\Users\PC\Desktop\Mohamed\IDM_VTON"
BASE_DIR  = r"C:\Users\PC\Desktop\Mohamed"
COMFY_DIR = os.path.join(BASE_DIR, "ComfyUI_windows_portable")
PY_PKGS   = os.path.join(COMFY_DIR, "python_embeded", "Lib", "site-packages")
AUX_SRC   = os.path.join(COMFY_DIR, "ComfyUI", "custom_nodes", "comfyui_controlnet_aux", "src")
MODELS    = os.path.join(COMFY_DIR, "ComfyUI", "models", "IDM-VTON")

for p in [PY_PKGS, APP_DIR, AUX_SRC,
          os.path.join(APP_DIR, "src"),
          os.path.join(APP_DIR, "preprocess"),
          os.path.join(APP_DIR, "preprocess", "openpose"),
          os.path.join(APP_DIR, "preprocess", "openpose", "annotator")]:
    sys.path.insert(0, p)

import annotator.util as au
au.annotator_ckpts_path = os.path.join(MODELS, "openpose", "ckpts")

import torch; print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
import onnxruntime as ort; print("onnxruntime:", ort.__version__)
import gradio as gr; print("gradio:", gr.__version__)
from transformers import CLIPTextModel; print("transformers: OK")
from diffusers import DDPMScheduler; print("diffusers: OK")
from src.unet_hacked_tryon import UNet2DConditionModel; print("src.unet_tryon: OK")
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DRef; print("src.unet_garmnet: OK")
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline; print("src.tryon_pipeline: OK")
from preprocess.openpose.run_openpose import OpenPose; print("openpose: OK")
from preprocess.humanparsing.parsing_api import onnx_inference; print("humanparsing: OK")
from utils_mask import get_mask_location; print("utils_mask: OK")

# Verify model files
for sub in ["unet", "vae", "text_encoder", "text_encoder_2",
            "image_encoder", "unet_encoder", "scheduler",
            "tokenizer", "tokenizer_2",
            "humanparsing", "openpose"]:
    path = os.path.join(MODELS, sub)
    print(f"  model/{sub}: {'OK' if os.path.exists(path) else 'MISSING'}")

print()
print("All imports and model files OK!")
