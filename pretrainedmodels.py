import torch
import torch.nn as nn
from transformers import BeitModel, SwinModel
from torchvision.models import resnet50
from PIL import Image
from transformers import BeitFeatureExtractor, SwinConfig, SwinModel

# Load Feature Extractors
beit_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")
swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
beit_model = BeitModel.from_pretrained("microsoft/beit-large-patch16-224")

# MedViT (Simulating with ResNet since MedViT is not in HuggingFace yet)
medvit_model = resnet50(pretrained=True)

class MultiModelFractureDetector(nn.Module):
    def __init__(self):
        super(MultiModelFractureDetector, self).__init__()
        
        # Extract feature dimensions
        self.swin_features = 768
        self.beit_features = 1024
        self.medvit_features = 2048  # ResNet50 last layer
        
        # Fusion Layer
        self.fc = nn.Linear(self.swin_features + self.beit_features + self.medvit_features, 512)
        self.classifier = nn.Linear(512, 2)  # Fracture or No Fracture

    def forward(self, swin_out, beit_out, medvit_out):
        # Concatenate features
        fused_features = torch.cat((swin_out, beit_out, medvit_out), dim=1)
        fused_features = self.fc(fused_features)
        output = self.classifier(fused_features)
        return output

# Initialize Model
model = MultiModelFractureDetector()

# Example Usage (X-ray Input)
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Process Image for Swin & BEiT
    inputs = beit_extractor(images=image, return_tensors="pt")
    swin_out = swin_model(**inputs).last_hidden_state.mean(dim=1)  # Swin features
    beit_out = beit_model(**inputs).last_hidden_state.mean(dim=1)  # BEiT features
    
    # Process Image for MedViT
    image_tensor = torch.rand(1, 3, 224, 224)  # Simulated input for MedViT
    medvit_out = medvit_model(image_tensor).unsqueeze(0)  # Get last layer output
    
    # Forward Pass
    result = model(swin_out, beit_out, medvit_out)
    return torch.sigmoid(result)

# Example X-ray
image_path = "fracture_xray.jpg"  # Replace with actual image
prediction = extract_features(image_path)
print("Fracture Prediction:", prediction)
