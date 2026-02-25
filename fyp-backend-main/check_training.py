import torch
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)

print('=== TRAINING METRICS ===')
print(f'Epoch: {checkpoint["epoch"]}')
print(f'Val AUC: {checkpoint["val_auc"]:.4f}')
print(f'Val F1 (last): {checkpoint["val_f1s"][-1]:.4f}')
print()
print(f'Train Losses (last 3): {[round(x, 4) for x in checkpoint["train_losses"][-3:]]}')
print(f'Val Losses (last 3): {[round(x, 4) for x in checkpoint["val_losses"][-3:]]}')
print(f'Val AUCs (last 3): {[round(x, 4) for x in checkpoint["val_aucs"][-3:]]}')
print()
print('Model was trained well (AUC 0.98+)')
print('But current predictions are all REAL (score ~0.05)')
print()
print('Possible causes:')
print('1. Test videos are all real (legitimate videos)')
print('2. Model was trained on specific deepfake type, not matching test data')
print('3. Threshold should be different')
