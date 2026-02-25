import torch
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']
keys = list(state_dict.keys())

print('=== CHECKPOINT STRUCTURE ===')
print(f'Total keys: {len(keys)}')
print(f'Epoch: {checkpoint["epoch"]}')
print(f'Val AUC: {checkpoint["val_auc"]:.4f}')

# Look for classifier keys
print('\n=== CLASSIFIER STRUCTURE ===')
classifier_keys = [k for k in keys if 'classifier' in k]
print(f'Classifier keys ({len(classifier_keys)}):')
for k in classifier_keys:
    shape = state_dict[k].shape
    print(f'  {k} {shape}')

# Look for temporal keys
print('\n=== TEMPORAL STRUCTURE ===')
temporal_keys = [k for k in keys if 'temporal' in k.lower() or 'conv3d' in k.lower()]
if temporal_keys:
    print(f'Found {len(temporal_keys)} temporal keys')
    for k in temporal_keys[:10]:
        shape = state_dict[k].shape
        print(f'  {k} {shape}')
else:
    print('No temporal/3D convolutions found!')

# List last 20 keys (likely head/output)
print('\n=== LAST 20 KEYS (Head/Output) ===')
for k in keys[-20:]:
    shape = state_dict[k].shape
    print(f'  {k} {shape}')
