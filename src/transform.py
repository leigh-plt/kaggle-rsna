from torchvision import transforms as T
from PIL import Image

train_transforms = T.Compose([
    Image.fromarray,
    T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0., hue=0.),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.345, 0.323, 0.271), std=(0.225, 0.324, 0.429))
    ])
    
inference_transforms = T.Compose([
    Image.fromarray,
    T.ToTensor(),
    T.Normalize(mean=(0.345, 0.323, 0.271), std=(0.225, 0.324, 0.429))
    ])