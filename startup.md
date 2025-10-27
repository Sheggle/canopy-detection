# Canopy Detection Server Setup Guide

## Prerequisites

### 1. Get SSH Details from vast.ai
Your vast.ai instance will provide an SSH command like:
```bash
ssh -p 53042 root@171.226.155.157 -L 8080:localhost:8080
```

From this example:
- `PORT` = `53042`
- `HOST` = `171.226.155.157`

**Replace `PORT` and `HOST` in all commands below with your actual values.**

### 2. Get Wandb API Key
Check your local `.env` file for:
```
WANDB_API_KEY=your_api_key_here
```

Or get it from: https://wandb.ai/authorize

## Quick Setup Command
```bash
ssh -p PORT root@HOST -L 8080:localhost:8080
```

## Complete Setup Process

### 1. Initial File Sync
**Replace `PORT` and `HOST` with your actual values from the SSH command**

Sync all necessary files to the remote server:

```bash
# Sync scripts directory
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" scripts/ root@HOST:~/scripts/

# Sync data files (zips and jsons only)
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" data/*.zip data/*.json root@HOST:~/data/

# Sync configuration files
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" pyproject.toml sweep_cv.yaml root@HOST:~/
```

### 2. Environment Setup
Connect to server and set up the Python environment:

```bash
# Connect to server
ssh -p PORT -o StrictHostKeyChecking=no root@HOST

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up Python environment and install dependencies
cd ~
export PATH=$HOME/.local/bin:$PATH
uv sync
```

### 3. Data Preparation
Unzip the data files:

```bash
cd data
unzip -q evaluation_images.zip -d evaluation_images
unzip -q train_images.zip -d train_images
```

### 4. Wandb Authentication
Login to wandb using the API key from .env:

```bash
export WANDB_API_KEY=YOUR_WANDB_API_KEY  # Get from local .env file
uv run python -c "import wandb; wandb.login()"
```

### 5. Verification
Verify the setup is complete:

```bash
# Check directory structure
ls -la
ls -la data/
ls -la scripts/

# Test key functionality
export PATH=$HOME/.local/bin:$PATH
uv run scripts/inference_yolo.py --help  # Should show help with -o parameter
```

## Key Files and Directories

### Project Structure
```
~/
├── pyproject.toml           # Python dependencies
├── sweep_cv.yaml           # Wandb sweep configuration
├── scripts/                # All Python scripts
│   ├── inference_yolo.py   # Updated with tqdm and -o parameter
│   ├── yolo_sweep_cv.py   # Updated with per-fold test submissions
│   ├── evaluate_v4.py     # Optimized evaluation (97x faster)
│   └── ...                # Other pipeline scripts
└── data/                  # Dataset
    ├── train_annotations.json
    ├── sample_answer.json
    ├── *.tif              # All training and evaluation images
    └── *.zip              # Original zip files (can be removed)
```

### Key Features Implemented
- **Per-fold test submissions**: Each CV fold generates `submission_fold_0.json`, `submission_fold_1.json`, etc.
- **Tqdm progress bars**: Inference shows progress during processing
- **Optimized evaluation**: 97x faster evaluation script (from 3min to 1.85s)
- **Custom scoring integration**: Wandb optimizes `custom_evaluation_score` instead of YOLO metrics

## Environment Details

### Dependencies Installed
- **PyTorch 2.9.0** with CUDA support
- **Ultralytics YOLO 8.3.221** for segmentation
- **OpenCV 4.12.0** for image processing
- **Shapely 2.1.2** for polygon operations
- **Wandb 0.22.2** for experiment tracking
- **tqdm** for progress bars
- **All other ML/CV libraries**

### GPU Support
- Full CUDA toolkit installed
- All nvidia-* packages for GPU acceleration
- MPS support for Apple Silicon (when available)

## Running Experiments

### Start a Wandb Sweep
```bash
cd ~
export PATH=$HOME/.local/bin:$PATH
export WANDB_API_KEY=YOUR_WANDB_API_KEY  # Get from .env file

# Create and run sweep
uv run wandb sweep sweep_cv.yaml
uv run wandb agent USERNAME/PROJECT/SWEEP_ID  # Use actual sweep ID from output
```

### Manual Training
```bash
# Create YOLO dataset
uv run scripts/to_yolo_cv.py 3  # 3-fold cross-validation

# Run cross-validation with per-fold submissions
uv run scripts/yolo_sweep_cv.py
```

### Inference Only
```bash
# Run inference on test set with progress bar
uv run scripts/inference_yolo.py model.pt -o my_submission.json
```

## Output Files

### Cross-Validation Results
- `experiments/cv_sweep_WANDB_RUN_ID/submission.json` - Combined CV submission
- `experiments/cv_sweep_WANDB_RUN_ID/submission_fold_0.json` - Fold 0 test submission
- `experiments/cv_sweep_WANDB_RUN_ID/submission_fold_1.json` - Fold 1 test submission
- `experiments/cv_sweep_WANDB_RUN_ID/submission_fold_2.json` - Fold 2 test submission
- ... (one for each fold configured in sweep_cv.yaml)

### Evaluation
- Use `scripts/evaluate_v4.py` for fast evaluation (97x speedup)
- Scores logged to wandb as `custom_evaluation_score`

## Troubleshooting

### Common Issues
1. **Permission denied**: Ensure SSH key is properly configured
2. **uv not found**: Run `export PATH=$HOME/.local/bin:$PATH`
3. **Missing dependencies**: Run `uv sync` to install all packages
4. **GPU issues**: Check `nvidia-smi` output and CUDA availability

### Performance Notes
- Evaluation takes ~1.85 seconds (down from 3+ minutes)
- Each fold training takes ~10-30 minutes depending on settings
- Test inference takes ~2-5 minutes with 150 images

## Environment Variables
```bash
export PATH=$HOME/.local/bin:$PATH
export WANDB_API_KEY=YOUR_WANDB_API_KEY  # Get from local .env file
export WANDB_AGENT_DISABLE_FLAPPING=true
```

## Notes
- Always use `uv run` prefix for Python commands
- The server has both training and evaluation images unzipped and ready
- Wandb is configured to optimize custom evaluation score instead of YOLO metrics
- Per-fold submissions enable multiple competition submissions for better leaderboard insights