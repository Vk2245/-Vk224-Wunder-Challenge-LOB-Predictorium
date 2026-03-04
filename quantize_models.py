"""
quantize_models.py — Convert ONNX models to INT8 for 3-4x speedup
==================================================================
INT8 quantization reduces model size and speeds up CPU inference
without significant accuracy loss.

RUN: python quantize_models.py
"""
import os
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_DIR = "models"
models_to_quantize = [
    "bigru.onnx",
    "timemixer.onnx",
    "tcn_t1.onnx",
]

print("="*60)
print("QUANTIZING MODELS TO INT8")
print("="*60)

for model_name in models_to_quantize:
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"⚠ Skipping {model_name} (not found)")
        continue
    
    # Output path
    base_name = model_name.replace(".onnx", "")
    quantized_path = os.path.join(MODEL_DIR, f"{base_name}_int8.onnx")
    
    print(f"\nQuantizing {model_name}...")
    
    # Get original size
    orig_size = os.path.getsize(model_path) / 1024 / 1024
    
    # Quantize
    try:
        quantize_dynamic(
            model_input=model_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,  # Quantize weights to 8-bit
        )
        
        # Get quantized size
        quant_size = os.path.getsize(quantized_path) / 1024 / 1024
        reduction = (1 - quant_size / orig_size) * 100
        
        print(f"  ✓ Original : {orig_size:.2f} MB")
        print(f"  ✓ Quantized: {quant_size:.2f} MB")
        print(f"  ✓ Reduction: {reduction:.1f}%")
        print(f"  ✓ Saved to : {quantized_path}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("QUANTIZATION COMPLETE")
print("="*60)
print("\nNEXT STEPS:")
print("1. Update solution.py to use *_int8.onnx models")
print("2. Test with test_solution.py")
print("3. Verify speed improvement (should be 3-4x faster)")
print("4. Check accuracy (should be within 1-2% of original)")
