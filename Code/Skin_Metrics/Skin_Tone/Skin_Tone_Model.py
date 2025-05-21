import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Image path
image_path = "Skin_Metrics/Skin_Tone/test_image.jpg" 

# class labels 
class_names = ['Black', 'Brown', 'White']

# Preprocessing 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model_path = "Skin_Metrics/Skin_Tone/skin_tone.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))
if hasattr(model, 'module'):
    model = model.module
model.eval()

if not os.path.exists(image_path):
    print(f"Error: Image path '{image_path}' not found.")
else:
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    print(f"Image: {image_path}")
    print(f"Predicted Skin Tone: {class_names[class_idx]}")

    # Show image with prediction
    plt.imshow(image)
    plt.title(f"Prediction: {class_names[class_idx]}")
    plt.axis('off')
    plt.show()
