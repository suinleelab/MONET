import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn

# import vit_shapley.modules.vision_transformer_verbose as vit
from torchvision import models as cnn_models

from MONET.models import classifier_utils


class ClassifierLitModule(pl.LightningModule):
    """`pytorch_lightning` module for image classifier.

    Args:
        backbone_type: should be the class name defined in `torchvision.models.cnn_models` or `timm.models.vision_transformer`
        download_weight: whether to initialize backbone with the pretrained weights
        load_path: If not None. loads the weights saved in the checkpoint to the model
        target_type: `binary` or `multi-class` or `multi-label`
        output_dim: the dimension of output
        checkpoint_metric: the metric used to determine whether to save the current status as checkpoints during the validation phase
        optim_type: type of optimizer for optimizing parameters
        learning_rate: learning rate of optimizer
        weight_decay: weight decay of optimizer
        decay_power: only `cosine` annealing scheduler is supported currently
        warmup_steps: parameter for the `cosine` annealing scheduler
    """

    def __init__(
        self,
        backbone_type: str,
        download_weight: bool,
        load_path: str or None,
        target_type: str,
        output_dim: int,
        checkpoint_metric: str or None,
        optim_type: str or None,
        learning_rate: float or None,
        loss_weight: None,
        weight_decay: float or None,
        decay_power: str or None,
        warmup_steps: int or None,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.logger_ = logging.getLogger(__name__)
        # print(self.hparams.download_weight, self.hparams.load_path)
        assert not (
            self.hparams.download_weight and self.hparams.load_path is not None
        ), "'download_weight' and 'load_path' cannot be activated at the same time as the downloaded weight will be overwritten by weights in 'load_path'."

        # Backbone initialization. (currently support only vit and cnn)
        if self.hparams.backbone_type == "resnet50":
            if self.hparams.download_weight:
                self.backbone = cnn_models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        else:
            raise NotImplementedError("Not supported backbone type")

        if self.hparams.download_weight:
            self.logger_.info(
                "The backbone parameters were initialized with the downloaded pretrained weights."
            )
        else:
            self.logger_.info("The backbone parameters were randomly initialized.")

        # Nullify classification head built in the backbone module and rebuild.
        if self.backbone.__class__.__name__ == "VisionTransformer":
            head_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif self.backbone.__class__.__name__ == "ResNet":
            head_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.head = nn.Linear(head_in_features, self.hparams.output_dim)
        elif self.backbone.__class__.__name__ == "DenseNet":
            head_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError("Not supported backbone type")

        # Set up modules for calculating metric
        classifier_utils.set_metrics(self, num_labels=self.hparams.output_dim)

    def configure_optimizers(self):
        return classifier_utils.set_schedule(self)

    def forward(self, images, output_attentions=False, output_hidden_states=False):
        if self.backbone.__class__.__name__ == "VisionTransformer":
            output = self.backbone(
                images,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            embedding_cls, embedding_tokens = output["x"], output["x_others"]
            # embedding_cls, embedding_tokens = self.backbone(images)
            logits = self.head(embedding_cls)
            output.update({"logits": logits})
        elif self.backbone.__class__.__name__ == "ResNet":
            out = self.backbone(images)
            logits = self.head(out)
            output = {"logits": logits}
        else:
            raise NotImplementedError("Not supported backbone type")

        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self(batch["image"])["logits"]

        return {"logits": logits, "label": labels}

    def training_step_end(self, batch_parts):
        loss = classifier_utils.compute_metrics(
            self,
            logits=batch_parts["logits"],
            labels=batch_parts["label"],
            phase="train",
        )

        return loss

    def training_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase="train")

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self(images)["logits"]

        return {"logits": logits, "label": labels}

    def validation_step_end(self, batch_parts):
        loss = classifier_utils.compute_metrics(
            self,
            logits=batch_parts["logits"],
            labels=batch_parts["label"],
            phase="val",
        )

    def validation_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase="val")

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self(images)["logits"]

        return {"logits": logits, "label": labels}

    def test_step_end(self, batch_parts):
        loss = classifier_utils.compute_metrics(
            self,
            logits=batch_parts["logits"],
            labels=batch_parts["label"],
            phase="test",
        )

    def test_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase="test")
