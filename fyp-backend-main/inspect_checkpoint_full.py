import torch
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)

print('=== CHECKPOINT CONTENTS ===\n')
for key in checkpoint.keys():
    if key not in ['model_state_dict', 'optimizer_state_dict']:
        value = checkpoint[key]
        if isinstance(value, (list, np.ndarray)):
            print(f'{key}: {type(value).__name__} with {len(value) if hasattr(value, "__len__") else "N/A"} items')
        else:
            print(f'{key}: {value}')

print('\n=== CLASS WEIGHTS ===')
print(checkpoint.get('class_weights', 'Not found'))

print('\n=== POSSIBLE PREPROCESSING INFO ===')
print('Note: If preprocessing information is not in the checkpoint,')
print('check the training script for normalization details.')
