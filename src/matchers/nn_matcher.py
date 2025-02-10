import numpy as np
import timm.data
import torch
from PIL import Image
from torchvision import transforms

from src.nn.embedding import EmbeddingModel


class NNMatcher:
    def __init__(self, encoder_name: str, embedding_size: int, weights_path: str, device: str = "cuda"):
        self.template_embedding = None
        self.sum_of_weight = []
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = EmbeddingModel(encoder_name, embedding_size).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
        ])

    def match_patches(self, patch):
        candidate_embedding = self.get_embedding(patch)

        distance = torch.dist(self.template_embedding, candidate_embedding, p=2).item()
        result = 1.0 - distance
        result = 0.5 * (result + 1.0)
        self.sum_of_weight.append(result)

        return result

    def compute_template(self, uav_patch):
        self.template_embedding = self.get_embedding(uav_patch)

    def get_sum_of_weight(self):
        sum_value = np.sum(self.sum_of_weight)
        self.sum_of_weight.clear()
        return sum_value

    def get_embedding(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        elif isinstance(img, Image.Image):
            img = self.transform(img)
        elif isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img = img / 255.0

        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img)

        return embedding.squeeze(0)
