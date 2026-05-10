"""
Lookzi Virtual Try-On — Standalone Web App
Run:  python app.py
Then open http://127.0.0.1:7860 in your browser
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# Reduce CUDA memory fragmentation — critical for fitting two large UNets on 8 GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(APP_DIR)
COMFY_DIR = os.path.join(BASE_DIR, "ComfyUI_windows_portable")
PY_PKGS   = os.path.join(COMFY_DIR, "python_embeded", "Lib", "site-packages")
MODELS    = os.path.join(COMFY_DIR, "ComfyUI", "models", "IDM-VTON")
AUX_SRC   = os.path.join(COMFY_DIR, "ComfyUI", "custom_nodes",
                          "comfyui_controlnet_aux", "src")

for p in [PY_PKGS, APP_DIR, AUX_SRC,
          os.path.join(APP_DIR, "src"),
          os.path.join(APP_DIR, "preprocess"),
          os.path.join(APP_DIR, "preprocess", "openpose"),
          os.path.join(APP_DIR, "preprocess", "openpose", "annotator")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Tell the OpenPose annotator where the weights live
import annotator.util as _au
_au.annotator_ckpts_path = os.path.join(MODELS, "openpose", "ckpts")

# ── Imports ────────────────────────────────────────────────────────────────────
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import gradio as gr

from transformers import (
    CLIPImageProcessor, CLIPVisionModelWithProjection,
    CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL

from src.unet_hacked_tryon   import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline      import StableDiffusionXLInpaintPipeline as TryonPipeline
# Note: src/ is from TemryL/ComfyUI-IDM-VTON (patched for diffusers 0.27.2)

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.parsing_api import onnx_inference
from utils_mask import get_mask_location

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
W, H   = 768, 1024


# ── Human Parsing (ONNX) ───────────────────────────────────────────────────────
class Parsing:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            os.path.join(MODELS, "humanparsing", "parsing_atr.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])
        self.lip_session = ort.InferenceSession(
            os.path.join(MODELS, "humanparsing", "parsing_lip.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])

    def __call__(self, img):
        return onnx_inference(self.session, self.lip_session, img)


# ── DensePose (TorchScript, no detectron2) ────────────────────────────────────
def load_densepose():
    cache_dir = os.path.join(COMFY_DIR, "ComfyUI", "models", "densepose_cache")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "densepose_r50_fpn_dl.torchscript")
    if not os.path.exists(model_path):
        print("Downloading DensePose TorchScript model (~200 MB, one time)...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="LayerNorm/DensePose-TorchScript-with-hint-image",
            filename="densepose_r50_fpn_dl.torchscript",
            local_dir=cache_dir,
        )
        print("DensePose downloaded.")
    # Load to CPU; run_tryon will move it to GPU only while it's needed.
    return torch.jit.load(model_path, map_location="cpu")


# ── Load everything ────────────────────────────────────────────────────────────
print("=" * 55)
print("  Lookzi Virtual Try-On — loading models...")
print("  (first run takes 2-3 minutes)")
print("=" * 55)

parsing_model  = Parsing()
openpose_model = OpenPose(0)
densepose_model = load_densepose()
densepose_model.eval()

unet = UNet2DConditionModel.from_pretrained(
    MODELS, subfolder="unet", torch_dtype=DTYPE).requires_grad_(False).eval()
unet_encoder = UNet2DConditionModel_ref.from_pretrained(
    MODELS, subfolder="unet_encoder", torch_dtype=DTYPE).requires_grad_(False).eval()
vae = AutoencoderKL.from_pretrained(
    MODELS, subfolder="vae", torch_dtype=DTYPE).requires_grad_(False).eval()
text_enc1 = CLIPTextModel.from_pretrained(
    MODELS, subfolder="text_encoder", torch_dtype=DTYPE).requires_grad_(False).eval()
text_enc2 = CLIPTextModelWithProjection.from_pretrained(
    MODELS, subfolder="text_encoder_2", torch_dtype=DTYPE).requires_grad_(False).eval()
img_enc = CLIPVisionModelWithProjection.from_pretrained(
    MODELS, subfolder="image_encoder", torch_dtype=DTYPE).requires_grad_(False).eval()
tok1  = AutoTokenizer.from_pretrained(MODELS, subfolder="tokenizer",  use_fast=False)
tok2  = AutoTokenizer.from_pretrained(MODELS, subfolder="tokenizer_2", use_fast=False)
sched = DDPMScheduler.from_pretrained(MODELS, subfolder="scheduler")

pipe = TryonPipeline.from_pretrained(
    MODELS,
    unet=unet, vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_enc1, text_encoder_2=text_enc2,
    tokenizer=tok1,  tokenizer_2=tok2,
    scheduler=sched, image_encoder=img_enc,
    torch_dtype=DTYPE,
)
pipe.unet_encoder = unet_encoder

# ── Memory optimisations ───────────────────────────────────────────────────────
# unet_encoder stays on CPU intentionally.  tryon_pipeline.py moves its outputs
# to GPU on-the-fly, so we never need both 3.5 GB UNets in VRAM simultaneously.
# Attention slicing + VAE slicing further cut peak VRAM during the denoising loop.
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()

tensor_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

print("All models ready.")


# ── DensePose helper ──────────────────────────────────────────────────────────
def get_pose_image(person_pil: Image.Image) -> Image.Image:
    """Run TorchScript DensePose → viridis-coloured body map."""
    import cv2
    from einops import rearrange

    img_rgb  = np.array(person_pil.resize((W, H)).convert("RGB"))
    inp      = rearrange(torch.from_numpy(img_rgb).float(), "h w c -> c h w")
    canvas   = np.zeros((H, W, 3), dtype=np.uint8)

    with torch.no_grad():
        pred_boxes, coarse_segm, fine_segm, u, v = densepose_model(inp.to(DEVICE))

    for i in range(len(pred_boxes)):
        x1, y1, x2, y2 = [int(c) for c in pred_boxes[i].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        parts = fine_segm[i].argmax(0).cpu().numpy().astype(np.uint8)
        parts = cv2.resize(parts, (bw, bh), interpolation=cv2.INTER_NEAREST)
        coloured = cv2.applyColorMap((parts * 10).clip(0, 255).astype(np.uint8),
                                     cv2.COLORMAP_VIRIDIS)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
        canvas[y1:y2, x1:x2] = coloured

    return Image.fromarray(canvas)


# ── Core try-on function ───────────────────────────────────────────────────────
def run_tryon(person_np, garment_np, garment_desc,
              auto_mask, steps, seed, progress=gr.Progress()):

    import gc

    if person_np is None or garment_np is None:
        return None, None, "Upload both a person photo and a garment photo."

    try:
        person  = Image.fromarray(person_np).convert("RGB").resize((W, H))
        garment = Image.fromarray(garment_np).convert("RGB").resize((W, H))

        # ── Mask (CPU only — OpenPose & ONNX parsing are fast on CPU) ─────────
        if auto_mask:
            progress(0.10, "OpenPose keypoint detection…")
            keypoints = openpose_model(person.resize((384, 512)))

            progress(0.20, "Human parsing…")
            parse_result, _ = parsing_model(person.resize((384, 512)))

            progress(0.30, "Building clothing mask…")
            mask, _ = get_mask_location("hd", "upper_body", parse_result, keypoints)
            mask = mask.resize((W, H))
        else:
            from PIL import ImageDraw
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).rectangle(
                [int(W * 0.05), int(H * 0.10), int(W * 0.95), int(H * 0.72)], fill=255)

        mask_gray_t  = (1 - tensor_tf(mask)) * tensor_tf(person)
        mask_preview = to_pil_image(((mask_gray_t + 1.0) / 2.0).clamp(0, 1))

        # ── DensePose: GPU only for this call, then immediately back to CPU ───
        progress(0.40, "DensePose body estimation…")
        densepose_model.to(DEVICE)
        pose_img = get_pose_image(person)
        densepose_model.to("cpu")
        torch.cuda.empty_cache()

        # ── Move pipeline to GPU (unet_encoder stays on CPU) ─────────────────
        # Peak VRAM breakdown:
        #   UNet 3.5 GB + VAE 0.3 GB + img_enc 0.3 GB = ~4.1 GB base
        #   + unet_encoder 3.5 GB (swapped in per step) → ~7.9 GB peak
        #   That fits inside an 8 GB card with slicing enabled.
        progress(0.50, "Moving pipeline to GPU…")
        pipe.to(DEVICE)

        # ── Encode text prompts, then free encoders — not needed in denoising ─
        progress(0.55, "Encoding prompts…")
        neg = "monochrome, lowres, bad anatomy, worst quality, low quality"
        with torch.no_grad():
            p_emb, n_emb, p_pool, n_pool = pipe.encode_prompt(
                f"model is wearing {garment_desc}",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=neg,
            )
            c_emb, _, _, _ = pipe.encode_prompt(
                [f"a photo of {garment_desc}"],
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=[""],
            )

        # Free ~0.9 GB VRAM — text encoders not used during denoising loop
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        # ── Diffusion ─────────────────────────────────────────────────────────
        progress(0.60, f"Running diffusion ({int(steps)} steps)…")
        pose_t    = tensor_tf(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
        garment_t = tensor_tf(garment).unsqueeze(0).to(DEVICE, DTYPE)
        gen = torch.Generator(DEVICE).manual_seed(int(seed))

        with torch.no_grad(), torch.cuda.amp.autocast(), torch.inference_mode():
            images = pipe(
                prompt_embeds=p_emb.to(DEVICE, DTYPE),
                negative_prompt_embeds=n_emb.to(DEVICE, DTYPE),
                pooled_prompt_embeds=p_pool.to(DEVICE, DTYPE),
                negative_pooled_prompt_embeds=n_pool.to(DEVICE, DTYPE),
                num_inference_steps=int(steps),
                generator=gen,
                strength=1.0,
                pose_img=pose_t,
                text_embeds_cloth=c_emb.to(DEVICE, DTYPE),
                cloth=garment_t,
                mask_image=mask,
                image=person,
                height=H, width=W,
                ip_adapter_image=garment,
                guidance_scale=2.0,
            )[0]

        return images[0], mask_preview, "Done!"

    except torch.cuda.OutOfMemoryError:
        return None, None, "GPU out of memory — try fewer denoising steps, then click Try It On again."
    except Exception as e:
        import traceback
        return None, None, f"Error: {e}\n{traceback.format_exc()}"
    finally:
        # Always clean up GPU memory — even after a crash — so the next run starts fresh.
        try:
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
            densepose_model.to("cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()


# ── Gradio UI ──────────────────────────────────────────────────────────────────
def _img_list(folder):
    if not os.path.exists(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])

HUMANS   = _img_list(os.path.join(APP_DIR,  "example", "human"))  + \
           _img_list(os.path.join(BASE_DIR, "Assets",  "Models"))
GARMENTS = _img_list(os.path.join(APP_DIR,  "example", "cloth"))  + \
           _img_list(os.path.join(BASE_DIR, "Assets",  "clothes"))

with gr.Blocks(title="Lookzi — Virtual Try-On", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👕 Lookzi — Virtual Try-On\nUpload a person photo and a garment, then click **Try It On**.")

    with gr.Row():
        with gr.Column():
            person_in = gr.Image(label="Person Photo", type="numpy", height=420)
            if HUMANS:
                gr.Examples(HUMANS, inputs=person_in, examples_per_page=5,
                            label="Sample people")

        with gr.Column():
            garment_in = gr.Image(label="Garment Photo", type="numpy", height=420)
            if GARMENTS:
                gr.Examples(GARMENTS, inputs=garment_in, examples_per_page=5,
                            label="Sample garments")

        with gr.Column():
            result_out  = gr.Image(label="Result", height=420)
            mask_out    = gr.Image(label="Detected clothing area", height=200)
            status_out  = gr.Textbox(label="Status", interactive=False, lines=1)

    desc_in = gr.Textbox(
        label="Garment description",
        placeholder="e.g. white cotton t-shirt, blue denim jacket",
        value="a shirt",
    )
    auto_mask_cb = gr.Checkbox(label="Auto-detect clothing area (recommended)", value=True)

    with gr.Accordion("Advanced settings", open=False):
        steps_sl = gr.Slider(15, 40, value=20, step=1, label="Denoising steps (20 = fast, 30 = better quality)")
        seed_nb  = gr.Number(value=42, label="Seed", precision=0)

    run_btn = gr.Button("✨ Try It On", variant="primary", size="lg")
    run_btn.click(
        fn=run_tryon,
        inputs=[person_in, garment_in, desc_in, auto_mask_cb, steps_sl, seed_nb],
        outputs=[result_out, mask_out, status_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860,
                share=False, inbrowser=True)
