from torchvision import transforms as T
from PIL import Image

train_transforms = T.Compose([
    Image.fromarray,
    T.ColorJitter(brightness=0.09, contrast=0.09, saturation=0., hue=0.),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.341, 0.162, 0.134), std=(0.312, 0.299, 0.269))
    ])
    
inference_transforms = T.Compose([
    Image.fromarray,
    T.ToTensor(),
    T.Normalize(mean=(0.341, 0.162, 0.134), std=(0.312, 0.299, 0.269))
    ])