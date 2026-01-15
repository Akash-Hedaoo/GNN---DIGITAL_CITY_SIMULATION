# Model Accuracy Improvements - Without Regenerating Training Data

## 🎯 Overview

This document explains the improvements made to increase model accuracy **without regenerating training data**. All improvements focus on:
1. Better hyperparameter tuning
2. Enhanced model architecture
3. Improved training techniques
4. Better regularization

---

## 📊 Key Improvements

### 1. **Architecture Enhancements**

#### Increased Model Capacity
- **Hidden Channels**: 64 → **96** (+50% capacity)
  - More parameters to learn complex patterns
  - Better representation learning
  
- **Attention Heads**: 4 → **6** (+50% attention capacity)
  - More attention mechanisms to capture different relationships
  - Better traffic flow pattern recognition

#### Added Regularization
- **Dropout**: 0.1 throughout the network
  - Prevents overfitting
  - Improves generalization
  
- **Layer Normalization**: Added after each convolution
  - Stabilizes training
  - Faster convergence
  - Better gradient flow

#### Residual Connections
- Added skip connections between GAT layers
  - Helps with gradient flow in deep networks
  - Allows model to learn identity mappings when needed
  - Prevents degradation in deeper layers

#### Enhanced Predictor Head
- **Deeper**: 2 layers → **3 layers** (128 → 64 → 1)
  - More capacity for final predictions
  - Better feature combination

---

### 2. **Training Improvements**

#### More Epochs
- **30 → 60 epochs** (+100%)
  - More time for model to converge
  - Better learning of complex patterns

#### Learning Rate Scheduling
- **ReduceLROnPlateau** scheduler
  - Automatically reduces LR when validation loss plateaus
  - Helps fine-tune model in later epochs
  - Prevents overshooting optimal solution

#### Better Optimizer
- **Adam → AdamW**
  - Improved weight decay implementation
  - Better generalization
  - More stable training

#### Weight Decay (L2 Regularization)
- **1e-5** weight decay
  - Prevents overfitting
  - Encourages smaller weights
  - Better generalization

#### Gradient Clipping
- Max gradient norm: **1.0**
  - Prevents exploding gradients
  - More stable training
  - Better convergence

#### Better Loss Function
- **MSE → Huber Loss** (delta=1.0)
  - More robust to outliers
  - Better handling of extreme values
  - Smoother gradients

#### Early Stopping
- **Patience: 10 epochs**
  - Stops training when validation loss stops improving
  - Prevents overfitting
  - Saves best model automatically

#### Better Data Splitting
- **Shuffled** before splitting
  - More representative train/val split
  - Better validation metrics

---

### 3. **Hyperparameter Changes**

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|------------|--------|
| **Epochs** | 30 | 60 | More training time |
| **Learning Rate** | 0.002 | 0.001 | More stable, with scheduler |
| **Hidden Channels** | 64 | 96 | More capacity |
| **Attention Heads** | 4 | 6 | Better attention |
| **Dropout** | None | 0.1 | Regularization |
| **Weight Decay** | None | 1e-5 | L2 regularization |
| **Loss Function** | MSE | Huber Loss | Robust to outliers |
| **Optimizer** | Adam | AdamW | Better weight decay |
| **Scheduler** | None | ReduceLROnPlateau | Adaptive LR |
| **Early Stopping** | None | Patience=10 | Prevent overfitting |

---

## 🔬 Expected Improvements

### Accuracy Gains
- **5-15% reduction** in validation loss expected
- **Better generalization** to unseen scenarios
- **More stable** predictions

### Training Behavior
- **Smoother convergence** curves
- **Less overfitting** due to regularization
- **Better final model** due to early stopping

### Model Characteristics
- **More robust** to outliers
- **Better feature learning** with deeper architecture
- **More stable** predictions

---

## 🚀 How to Use

### Option 1: Train Improved Model
```powershell
python step4_train_model_improved.py
```

This will:
1. Load existing training data (`gnn_training_data.pkl`)
2. Train improved model with all enhancements
3. Save to `real_city_gnn_improved.pt`
4. Show training progress with best model tracking

### Option 2: Compare Results
After training, compare:
- Old model: `real_city_gnn.pt`
- New model: `real_city_gnn_improved.pt`

Use validation loss as the metric.

---

## 📈 Monitoring Training

Watch for:
- **Validation loss decreasing** steadily
- **Learning rate reduction** when plateau detected
- **Early stopping** if validation loss stops improving
- **Best model** automatically saved

---

## ⚠️ Memory Considerations

The improved model uses:
- **~20-30% more VRAM** (due to larger hidden channels)
- Still fits in **6GB GPU** with batch size 8
- If OOM occurs, reduce `HIDDEN_CHANNELS` to 80 or `BATCH_SIZE` to 6

---

## 🎯 Next Steps After Training

1. **Evaluate** on validation set
2. **Compare** with old model
3. **Test** on what-if scenarios (`step5_whatif_analysis.py`)
4. **Deploy** if improvements are significant

---

## 💡 Additional Improvements (Future)

If you want even more accuracy:

1. **Ensemble Models**: Train multiple models and average predictions
2. **Hyperparameter Search**: Grid search for optimal LR, dropout, etc.
3. **Cross-Validation**: Better validation strategy
4. **Feature Engineering**: Add more node/edge features
5. **Data Augmentation**: Add noise, rotations (limited without regenerating)

---

*Created: January 13, 2026*
*Improvements: Architecture, Training, Regularization*
