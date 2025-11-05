import torch
from torchvision import transforms
from PIL import Image
import json
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feed_count_model.pth')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.json')

_model = None


# Load the correct model architecture and weights
def get_model(model_path=None, device=None):
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_path is None:
        # Default path (update as needed)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoint', 'best_optimized_epoch_79.pth')
    # Try to import enhanced model first
    try:
        from enhanced_mcnn_model import EnhancedMCNNForPellets
        model = EnhancedMCNNForPellets().to(device)
    except ImportError:
        from mcnn_model import ImprovedMCNN
        model = ImprovedMCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# Predict pellet count using the actual model output (sum of density map)
def predict_pellets(model, image_file, device=None):
    from PIL import Image
    import numpy as np
    import torch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(image_file).convert('RGB')
    # Model expects 512x512 input, normalized to [0,1]
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        # Output is a density map, sum to get count
        pellet_count = float(output.sum().item())
    return pellet_count

def get_feed_ratio():
    if not os.path.exists(CONFIG_PATH):
        return {'pellets': 50, 'grams': 10}
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def set_feed_ratio(pellets, grams):
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'pellets': pellets, 'grams': grams}, f)
