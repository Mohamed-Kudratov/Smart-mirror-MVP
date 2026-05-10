"""Quick smoke test: run OpenPose + human parsing on a sample image."""
import sys, os, warnings
warnings.filterwarnings("ignore")

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(APP_DIR)
COMFY_DIR = os.path.join(BASE_DIR, "ComfyUI_windows_portable")
PY_PKGS   = os.path.join(COMFY_DIR, "python_embeded", "Lib", "site-packages")
MODELS    = os.path.join(COMFY_DIR, "ComfyUI", "models", "IDM-VTON")

for p in [PY_PKGS, APP_DIR,
          os.path.join(APP_DIR, "preprocess"),
          os.path.join(APP_DIR, "preprocess", "openpose"),
          os.path.join(APP_DIR, "preprocess", "openpose", "annotator")]:
    sys.path.insert(0, p)

import annotator.util as au
au.annotator_ckpts_path = os.path.join(MODELS, "openpose", "ckpts")

import onnxruntime as ort
from PIL import Image
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.parsing_api import onnx_inference
from utils_mask import get_mask_location

# Find a sample image
person_path = os.path.join(BASE_DIR, "Assets", "Models", "00034_00.jpg")
assert os.path.exists(person_path), f"Sample image not found: {person_path}"

print("Loading sample image:", person_path)
person = Image.open(person_path).convert("RGB")
small  = person.resize((384, 512))

print("Running OpenPose...")
openpose = OpenPose(0)
keypoints = openpose(small)
print(f"  Keypoints: {len(keypoints['pose_keypoints_2d'])} joints detected")

print("Running Human Parsing (ONNX)...")
opts = ort.SessionOptions()
sess     = ort.InferenceSession(os.path.join(MODELS, "humanparsing", "parsing_atr.onnx"),
                                sess_options=opts, providers=["CPUExecutionProvider"])
lip_sess = ort.InferenceSession(os.path.join(MODELS, "humanparsing", "parsing_lip.onnx"),
                                sess_options=opts, providers=["CPUExecutionProvider"])
parse_result, _ = onnx_inference(sess, lip_sess, small)
print(f"  Parse result size: {parse_result.size}")

print("Generating clothing mask...")
mask, mask_gray = get_mask_location("hd", "upper_body", parse_result, keypoints)
mask = mask.resize((768, 1024))
mask.save("test_mask_output.png")
print(f"  Mask saved to: test_mask_output.png")

print()
print("Preprocessing test PASSED!")
