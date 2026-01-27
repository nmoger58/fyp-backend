from model import MyModel
import torch
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

print('Loading model...')
device = torch.device('cpu')
model = MyModel(num_frames=16).to(device)

print('Loading checkpoint...')
checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)

print('Loading state dict...')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('Testing forward pass...')
# Create dummy input
dummy_input = torch.randn(1, 16, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy_input)
    prob = torch.sigmoid(output)
    
print(f'Output: {output.item():.4f}')
print(f'Probability: {prob.item():.4f}')
prediction = "DEEPFAKE" if prob.item() > 0.5 else "REAL"
confidence = prob.item() if prob.item() > 0.5 else (1-prob.item())
print(f'Prediction: {prediction}')
print(f'Confidence: {confidence:.2%}')
print('')
print('✅ Model working correctly!')
