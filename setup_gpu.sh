#!/bin/bash
# ============================================================
# setup_gpu.sh — One-shot setup for GPU rental machine
# Run this once after uploading your project zip.
#
# Usage:
#   chmod +x setup_gpu.sh
#   ./setup_gpu.sh
# ============================================================

set -e  # exit on any error

echo "============================================"
echo " FMO Surrogate — GPU Environment Setup"
echo "============================================"

# 1. Check GPU
nvidia-smi
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# 2. Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    qutip \
    h5py \
    numpy scipy matplotlib \
    scikit-learn \
    tqdm \
    pyyaml \
    -q

echo "All packages installed."

# 3. Copy GPU-optimised files into place
echo ""
echo "Applying GPU-optimised source files..."
cp config_gpu.yaml config.yaml
cp generate_data_gpu.py src/generate_data.py
cp train_gpu.py src/train.py

# 4. Create directories
mkdir -p data results/checkpoints results/logs results/figures src/analysis

# 5. Final check
python3 -c "
import torch, qutip, h5py, numpy, scipy, sklearn
print('All imports OK')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')
"

echo ""
echo "============================================"
echo " Setup complete. Run order:"
echo ""
echo "  # 1. Generate Lindblad data (parallel, ~15 mins)"
echo "  python src/generate_data.py --config config.yaml --mode lindblad --workers 16"
echo ""
echo "  # 2. Generate HEOM data (serial, ~30 mins)"
echo "  python src/generate_data.py --config config.yaml --mode heom"
echo ""
echo "  # 3. Train all models (all 4, ~45 mins total)"
echo "  python src/train.py --model all --config config.yaml"
echo ""
echo "  # 4. Evaluate"
echo "  python src/evaluate.py --config config.yaml"
echo ""
echo "  # 5. New analyses"
echo "  python src/run_analysis.py --config config.yaml"
echo "============================================"
