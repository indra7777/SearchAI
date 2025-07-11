# Budget Training Guide: Under $10 on RunPod

This guide ensures your Visual Intelligence Pipeline training stays under $10 on RunPod.

## Cost Analysis & Strategy

### RunPod Pricing (as of 2024)
- **RTX 4090**: $0.50-0.79/hour (on-demand)
- **RTX 4090**: $0.25-0.45/hour (spot instances)
- **RTX 3090**: $0.35-0.55/hour (on-demand)
- **RTX 3090**: $0.20-0.35/hour (spot instances)

### Budget Breakdown for $10
| GPU | Type | Hours Available | Strategy |
|-----|------|----------------|----------|
| RTX 4090 | Spot | 22-40 hours | Recommended |
| RTX 4090 | On-demand | 12-20 hours | Backup |
| RTX 3090 | Spot | 28-50 hours | Budget option |

## Strategy 1: Lightweight Training (Recommended)

### Reduced Dataset Size
Instead of full datasets, use smaller subsets:
- **COCO**: 10,000 images (vs 120,000)
- **Places365**: 5,000 images (vs 1.8M)
- **DTD**: Full dataset (5,640 images - already small)

### Fast Training Configuration
```json
{
  "training": {
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 3e-4,
    "early_stopping_patience": 8
  },
  "model_config": {
    "backbone": "efficientnet-b0",
    "input_size": 416,
    "reduced_complexity": true
  }
}
```

**Estimated Time**: 6-10 hours
**Estimated Cost**: $3-7 (spot) / $5-12 (on-demand)

## Strategy 2: Progressive Training

### Phase 1: Core Components (4-6 hours)
1. Object Detection only
2. Scene Classification only
3. Basic evaluation

### Phase 2: Multi-task (4-6 hours)
1. Add relationship analysis
2. Add facial analysis
3. Fine-tune ensemble

**Total Estimated Cost**: $4-8

## Implementation: Budget-Optimized Setup
