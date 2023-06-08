import torch
from torch import nn
from torchvision import models as cnn_models

# efficientnet = torchvision.models.efficientnet_v2_s(
#     weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
# ).to(device)
# efficientnet.eval()
import clip


class ImageClassifier(nn.Module):
    def __init__(self, backbone_type, output_dim):
        super().__init__()
        self.backbone_type = backbone_type

        if self.backbone_type == "efficientnet_v2_s":

            self.backbone = cnn_models.efficientnet_v2_s(
                weights=cnn_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            )
            self.in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.in_features, output_dim))
        elif self.backbone_type == "ViT-B/32":
            clip_model = clip.load("ViT-B/32", jit=False)[0]
            for params in clip_model.parameters():
                params.requires_grad = False
            self.backbone = clip_model
            self.in_features = 512
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.in_features, output_dim))

    def forward(self, x):
        if self.backbone_type == "efficientnet_v2_s":
            out = self.backbone(x)
            logits = self.head(out)
            return logits
        elif self.backbone_type == "ViT-B/32":
            image_features = self.backbone.encode_image(x)
            logits = self.head(image_features.float())
            return logits


if __name__ == "__main__":
    net = ImageClassifier(backbone_type="ViT-B/32", output_dim=48)

    net(torch.randn(10, 3, 224, 224))
