from torch import nn
from transformers import AutoModel

import clip


class ImageTextEncoder(nn.Module):
    def __init__(
        self,
        backbone_api: str,
        model_name_or_path: str,
        graident_checkpoiting_segments: int,
        device: str,
    ):
        super().__init__()

        self.backbone_api = backbone_api
        self.model_name_or_path = model_name_or_path

        if backbone_api == "clip":
            self.model, self.preprocess = clip.load(model_name_or_path, device=device, jit=False)
            assert self.model.transformer.graident_checkpoiting_segments == 0
            self.model.transformer.graident_checkpoiting_segments = graident_checkpoiting_segments

        elif backbone_api == "huggingface":
            self.model = AutoModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError("backbone_api must be either clip or huggingface")

    def encode_image(self, image):
        if self.backbone_api == "clip":
            return self.model.encode_image(image)
        elif self.backbone_api == "huggingface":
            return self.model.get_image_features(image)
        else:
            raise ValueError("backbone_api must be either clip or huggingface")

    def encode_text(self, text):
        if self.backbone_api == "clip":
            return self.model.encode_text(text)
        elif self.backbone_api == "huggingface":
            return self.model.get_text_features(text)
        else:
            raise ValueError("backbone_api must be either clip or huggingface")

    def forward(self, image, text):
        if self.backbone_api == "clip":
            return self.model(image, text)
        elif self.hparams.backbone_api == "huggingface":
            return self.model(image, text)
        else:
            raise ValueError("backbone_api must be either clip or huggingface")


if __name__ == "__main__":
    image_text_encoder_clip = ImageTextEncoder(backbone_api="clip", model_name_or_path="ViT-B/32")
    image_text_encoder_hf = ImageTextEncoder(
        backbone_api="huggingface",
        model_name_or_path="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    )
    print(image_text_encoder_clip)
    print(image_text_encoder_hf)
