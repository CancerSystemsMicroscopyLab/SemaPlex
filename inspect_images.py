"""
Quick image inspection script - understand your input/output images.
No matplotlib required - just prints statistics and saves annotated images.
"""
import os
import cv2
import numpy as np

def analyze_image(img, name):
    """Print detailed statistics about an image."""
    print(f"\n{'─'*50}")
    print(f"📊 {name}")
    print(f"{'─'*50}")
    print(f"  Shape:        {img.shape}")
    print(f"  Data type:    {img.dtype}")
    print(f"  Value range:  [{img.min():,} → {img.max():,}]")
    print(f"  Mean:         {img.mean():.2f}")
    print(f"  Std dev:      {img.std():.2f}")
    print(f"  Median:       {np.median(img):.2f}")
    
    # Percentiles
    p10, p50, p90 = np.percentile(img, [10, 50, 90])
    print(f"  Percentiles:  10%={p10:.1f}, 50%={p50:.1f}, 90%={p90:.1f}")
    
    # Non-zero pixels
    nonzero = np.count_nonzero(img)
    total = img.size
    print(f"  Non-zero:     {nonzero:,} / {total:,} ({100*nonzero/total:.1f}%)")
    
    # Histogram bins
    hist, bins = np.histogram(img.flatten(), bins=10)
    print("  Distribution:")
    for i, (count, low, high) in enumerate(zip(hist, bins[:-1], bins[1:])):
        bar = '█' * int(40 * count / hist.max())
        print(f"    [{low:7.1f}-{high:7.1f}]: {bar} ({count:,})")

def inspect_sample(sample_name, dataset_root, results_root):
    """Inspect one sample across all channels."""
    
    print(f"\n{'='*70}")
    print(f"🔬 SAMPLE: {sample_name}")
    print(f"{'='*70}")
    
    # Load images
    mix555_path = os.path.join(dataset_root, 'mix555', f'{sample_name}.tiff')
    dapi_path = os.path.join(dataset_root, 'DAPI', f'{sample_name}.tiff')
    lamin_path = os.path.join(dataset_root, 'lamin', f'{sample_name}.tiff')
    fake_path = os.path.join(results_root, 'images', f'{sample_name}_fake0.tif')
    
    mix555 = cv2.imread(mix555_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    dapi = cv2.imread(dapi_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    lamin = cv2.imread(lamin_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE) if os.path.exists(lamin_path) else None
    fake = cv2.imread(fake_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(fake_path) else None
    
    if mix555 is None:
        print(f"❌ Could not load input images for {sample_name}")
        return
    
    # Analyze each channel
    analyze_image(mix555, "INPUT 0: mix555 (mixed fluorescence)")
    analyze_image(dapi, "INPUT 1: DAPI (nuclear stain)")
    
    if lamin is not None:
        analyze_image(lamin, "TARGET: lamin (ground truth)")
    else:
        print("\n⚠️  No ground truth (lamin) available for this sample")
    
    if fake is not None:
        analyze_image(fake, "PREDICTION: fake (model output)")
        
        if lamin is not None:
            # Normalize both to same scale for comparison
            lamin_norm = lamin.astype(float)
            fake_norm = fake.astype(float)
            
            mae = np.mean(np.abs(lamin_norm - fake_norm))
            mse = np.mean((lamin_norm - fake_norm) ** 2)
            
            print(f"\n{'─'*50}")
            print("📏 PREDICTION ACCURACY")
            print(f"{'─'*50}")
            print(f"  Mean Absolute Error:  {mae:.2f}")
            print(f"  Mean Squared Error:   {mse:.2f}")
            print(f"  RMSE:                 {np.sqrt(mse):.2f}")
            
            if lamin.max() > 0:
                normalized_mae = mae / lamin.max()
                print(f"  Normalized MAE:       {normalized_mae:.4f} ({normalized_mae*100:.2f}% of max)")
    else:
        print("\n⚠️  No prediction available - run testing first")

def main():
    """Main inspection routine."""
    dataset_root = r'C:\Arnas\PythonWork\SemaPlex\sample_dataset'
    results_root = r'C:\Arnas\PythonWork\SemaPlex\results\mix555_DAPI\fold0\unet\test_results'
    
    print("\n" + "="*70)
    print("🔬 IMAGE UNDERSTANDING GUIDE")
    print("="*70)
    print("""
WHAT THIS SCRIPT DOES:
Analyzes the input/output images from your fluorescence unmixing model.

YOUR TASK:
Input:  2 channels (mix555 + DAPI) → Model → Output: 1 channel (lamin)

CHANNELS EXPLAINED:
  🟡 mix555: Mixed fluorescence signal containing multiple markers
             (Think: a cocktail of fluorescent signals all captured together)
  
  🔵 DAPI:   Nuclear stain - highlights cell nuclei
             (Think: GPS coordinates showing where cells are)
  
  🟢 lamin:  Target protein at nuclear envelope
             (Think: the specific signal we want to extract from the mix)

MODEL'S JOB:
Learn to "unmix" the mixed signal to extract just the lamin component.
Like a bartender identifying one ingredient in a cocktail by taste.

IMAGE NOTES:
  - All are grayscale TIFF images (single channel)
  - Higher intensity = more of that fluorescent marker present
  - Size: varies in original, cropped to 256×256 during training
""")
    print("="*70 + "\n")
    
    # List available samples
    mix555_dir = os.path.join(dataset_root, 'mix555')
    if os.path.exists(mix555_dir):
        samples = sorted([f.replace('.tiff', '') for f in os.listdir(mix555_dir) if f.endswith('.tiff')])
        print(f"Found {len(samples)} samples in dataset")
        print(f"Training on: {len([s for s in samples if os.path.exists(os.path.join(dataset_root, 'lamin', s+'.tiff'))])} samples with ground truth")
    else:
        print("❌ Dataset not found!")
        return
    
    # Inspect a few representative samples
    inspect_samples = samples[:3]  # First 3 samples
    
    for sample_name in inspect_samples:
        try:
            inspect_sample(sample_name, dataset_root, results_root)
        except Exception as e:
            print(f"\n❌ Error inspecting {sample_name}: {e}")
    
    print("\n" + "="*70)
    print("💡 INTERPRETATION TIPS")
    print("="*70)
    print("""
WHAT TO LOOK FOR:

1. INPUT BRIGHTNESS:
   - If inputs are very dim (mean < 10), model has little signal to work with
   - If inputs are saturated (many pixels at max), data may be clipped

2. TARGET vs PREDICTION:
   - Compare mean/std between ground truth and prediction
   - If prediction mean is much lower, model is being too conservative
   - If MAE is high, model needs more training or better data

3. YOUR CURRENT STATUS:
   - Prediction mean: ~8-13 (quite dim)
   - Prediction max:  ~110-166 (out of 255)
   - This suggests model is learning patterns but may be under-confident

4. NEXT STEPS:
   ✓ If ground truth is bright (mean > 50): increase training epochs
   ✓ If ground truth is dim too (mean < 20): model is doing okay!
   ✓ If MAE is high: try lower lambda_A or more training data
   ✓ Compare non-zero percentages between target and prediction
""")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
