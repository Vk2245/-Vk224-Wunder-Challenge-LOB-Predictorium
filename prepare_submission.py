"""
prepare_submission.py — Run from project root AFTER test_solution.py passes.
Packages solution.py + 2-MODEL ENSEMBLE (bigru.onnx + timemixer.onnx + scalers) into submission.zip
python prepare_submission.py
"""
import os, zipfile, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

FILES = {
    os.path.join(ROOT, "src", "solution_2model.py"): "solution.py",
    os.path.join(ROOT, "models", "bigru_int8.onnx"):    "bigru_int8.onnx",
    os.path.join(ROOT, "models", "bigru.onnx"):    "bigru.onnx",
    os.path.join(ROOT, "models", "scaler.npz"):    "scaler.npz",
    os.path.join(ROOT, "models", "timemixer_int8.onnx"): "timemixer_int8.onnx",
    os.path.join(ROOT, "models", "timemixer.onnx"): "timemixer.onnx",
    os.path.join(ROOT, "models", "scaler_tm.npz"): "scaler_tm.npz",
}

OUT = os.path.join(ROOT, "submission.zip")

print("Checking files...")
for src, dst in FILES.items():
    if not os.path.exists(src):
        print(f"  MISSING: {src}")
        print("  Cannot build submission. Run train_bigru.py first.")
        exit(1)
    mb = os.path.getsize(src) / 1024 / 1024
    print(f"  OK  {dst}  ({mb:.2f} MB)")

total_mb = sum(os.path.getsize(s) for s in FILES) / 1024 / 1024
print(f"\nTotal size: {total_mb:.2f} MB")
if total_mb > 50:
    print("WARNING: zip may exceed competition limit")

print(f"\nBuilding {OUT} ...")
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for src, dst in FILES.items():
        zf.write(src, dst)
        print(f"  Added: {dst}")

zip_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"\nsubmission.zip ready: {zip_mb:.2f} MB")
print("Upload to: https://wundernn.io/predictorium/submit")