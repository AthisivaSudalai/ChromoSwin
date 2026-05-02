"""
Swin Transformer + HMFO Optimization Training with Checkpoint Support for Kaggle

Features:
- Phase 1: HMFO hyperparameter optimization
- Phase 2: Full training with optimized hyperparameters
- Saves checkpoints every N epochs
- Can resume from last checkpoint
- Handles session interruptions
- Saves best model automatically
"""

from src.dataset import get_dataloaders, get_class_weights
from src.swin_model import build_swin
from src.train import train_model
from src.hmfo import run_hmfo
import os
import torch
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (adjust for Kaggle)
DATA_DIR = '/kaggle/working/data_preprocessed'  # Kaggle dataset location
OUTPUT_DIR = '/kaggle/working'  # Kaggle working directory (persists between sessions)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
SAVE_EVERY = 5  # Save checkpoint every 5 epochs

# HMFO settings
N_MOTHS = 20  # Number of moths in HMFO
N_ITERATIONS = 30  # Number of HMFO iterations

# Resume from checkpoint?
RESUME = True  # Set to True to resume from last checkpoint

# ============================================================================
# SETUP
# ============================================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'results/swin_hmfo'), exist_ok=True)

print("="*70)
print("SWIN TRANSFORMER + HMFO OPTIMIZATION (KAGGLE)")
print("="*70)
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print(f"Resume from checkpoint: {RESUME}")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=DATA_DIR, 
    batch_size=BATCH_SIZE
)

class_weights = get_class_weights(DATA_DIR)

print(f"✓ Data loaded")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# PHASE 1: HMFO HYPERPARAMETER OPTIMIZATION
# ============================================================================

best_params_path = os.path.join(OUTPUT_DIR, 'results/swin_hmfo/best_params.json')
hmfo_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'hmfo_optimization.json')

# Check if HMFO optimization already completed
if RESUME and os.path.exists(best_params_path):
    print("\n" + "="*70)
    print("LOADING HMFO RESULTS")
    print("="*70)
    
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    print(f"✓ Loaded HMFO-optimized hyperparameters:")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Batch size: {best_params.get('batch_size', BATCH_SIZE)}")
    print(f"  Best F1 from HMFO: {best_params.get('best_f1', 'N/A')}")
    print("="*70)
    
else:
    print("\n" + "="*70)
    print("PHASE 1: HMFO HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Moths: {N_MOTHS}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"This will take ~{N_MOTHS * N_ITERATIONS * 2 // 60} minutes")
    print("="*70)
    
    best_params, best_f1, hmfo_history = run_hmfo(
        train_loader, val_loader,
        n_moths=N_MOTHS,
        n_iterations=N_ITERATIONS,
        num_classes=24
    )
    
    # Save HMFO results
    best_params['best_f1'] = best_f1
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    with open(hmfo_checkpoint_path, 'w') as f:
        json.dump({'best_params': best_params, 'history': hmfo_history}, f, indent=2)
    
    print("\n" + "="*70)
    print("HMFO OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"✓ Best hyperparameters found:")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"✓ Saved to: {best_params_path}")
    print("="*70)

# ============================================================================
# PHASE 2: FULL TRAINING WITH OPTIMIZED HYPERPARAMETERS
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: FULL TRAINING WITH HMFO-OPTIMIZED HYPERPARAMETERS")
print("="*70)

LEARNING_RATE = best_params['learning_rate']
print(f"Using learning rate: {LEARNING_RATE}")

# ============================================================================
# BUILD MODEL
# ============================================================================

print("\nBuilding model...")
model = build_swin(num_classes=24, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"✓ Model built and moved to {device}")

# ============================================================================
# RESUME FROM CHECKPOINT (if exists)
# ============================================================================

start_epoch = 0
best_f1 = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

checkpoint_path = os.path.join(CHECKPOINT_DIR, 'hmfo_latest_checkpoint.pth')

if RESUME and os.path.exists(checkpoint_path):
    print("\n" + "="*70)
    print("RESUMING FROM CHECKPOINT")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint.get('best_f1', 0.0)
    history = checkpoint.get('history', history)
    
    print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    print(f"  Best F1 so far: {best_f1:.4f}")
    print(f"  Continuing from epoch {start_epoch}")
    print("="*70)
else:
    print("\n✓ Starting training from scratch")

# ============================================================================
# TRAINING LOOP WITH CHECKPOINTS
# ============================================================================

print("\n" + "="*70)
print("TRAINING")
print("="*70)

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(start_epoch, EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 70)
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_path = os.path.join(OUTPUT_DIR, 'results/swin_hmfo/best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Saved best model (F1: {best_f1:.4f})")
    
    # Save checkpoint every N epochs
    if (epoch + 1) % SAVE_EVERY == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'history': history,
            'hmfo_params': best_params
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Also save history after each epoch
    history_path = os.path.join(OUTPUT_DIR, 'results/swin_hmfo/history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

# ============================================================================
# FINAL SAVE
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, 'results/swin_hmfo/final_model.pth')
torch.save(model.state_dict(), final_model_path)

# Save final checkpoint
final_checkpoint = {
    'epoch': EPOCHS - 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_f1': best_f1,
    'history': history,
    'hmfo_params': best_params
}
torch.save(final_checkpoint, checkpoint_path)

# Save final summary
summary = {
    'hmfo_params': best_params,
    'best_f1': best_f1,
    'final_epoch': EPOCHS,
    'training_complete': True
}
summary_path = os.path.join(OUTPUT_DIR, 'results/swin_hmfo/training_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Training complete!")
print(f"  Best F1: {best_f1:.4f}")
print(f"  HMFO learning rate: {best_params['learning_rate']}")
print(f"  Final model saved to: {final_model_path}")
print(f"  Best model saved to: {best_model_path}")
print(f"  History saved to: {history_path}")
print(f"  Summary saved to: {summary_path}")
print("="*70)
