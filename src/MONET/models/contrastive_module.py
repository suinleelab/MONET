from collections import OrderedDict
from typing import Any, List

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

from MONET.utils.metrics import skincon_calcualte_auc_all
from MONET.utils.text_processing import (
    generate_prompt_token_from_caption,
    generate_prompt_token_from_concept,
)


class ContrastiveLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_mode="text",
        val_mode="text",
        test_mode="text",
        automatic_optimization=True,
    ):
        super().__init__()
        if not automatic_optimization:
            self.automatic_optimization = False
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # for p in self.net.parameters():
        #     # print(p.data.dtype, p.grad.data.dtype)
        #     p.data = p.data.float()
        #         p.grad.data = p.grad.data.float()

        # loss function
        self.criterion_img = torch.nn.CrossEntropyLoss()
        self.criterion_txt = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_loss = MeanMetric()
        self.train_metadata = OrderedDict()
        self.train_text_features = OrderedDict()

        self.val_loss = MeanMetric()
        self.val_metadata = OrderedDict()
        self.val_text_features = OrderedDict()

        self.test_loss = MeanMetric()
        self.test_metadata = OrderedDict()
        self.test_text_features = OrderedDict()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_auc_best = MaxMetric()

    def forward(self, image, text):
        return self.net(image, text)

    def model_step_with_image_text(self, batch: Any):
        if "text" not in batch:
            print(batch.keys())
            raise ValueError("Batch must contain 'text' key")
        image, text = batch["image"], batch["text"]
        image_features, text_features = self.net.encode_image(image), self.net.encode_text(text)
        return {"image_features": image_features, "text_features": text_features}

    def model_step_with_image(self, batch: Any):
        image = batch["image"]
        image_features = self.net.encode_image(image)
        return {"image_features": image_features}

    def model_step_with_text(self, batch: Any):
        text = batch["text"]
        text_features = self.net.encode_text(text)
        return {"text_features": text_features}

    def features_to_logits(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.net.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }

    def logits_to_loss(self, logits_per_image, logits_per_text):
        ground_truth = torch.arange(
            len(logits_per_image), dtype=torch.long, device=logits_per_image.device
        )
        loss = (
            self.criterion_img(logits_per_image, ground_truth)
            + self.criterion_txt(logits_per_text, ground_truth)
        ) / 2

        return loss

    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     for p in self.net.parameters():
    #         # print(p.data.dtype, p.grad.data.dtype)
    #         p.data = p.data.float()
    #         p.grad.data = p.grad.data.float()

    #     # print(optimizer.param_groups[0]["lr"])
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     clip.model.convert_weights(model)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_auc_best.reset()
        self.val_loss_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        return self.model_step_with_image_text(batch)

    def training_step_end(self, batch_parts):
        logits = self.features_to_logits(
            image_features=batch_parts["image_features"],
            text_features=batch_parts["text_features"],
        )
        loss = self.logits_to_loss(
            logits_per_image=logits["logits_per_image"],
            logits_per_text=logits["logits_per_text"],
        )
        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # # we can return here dict with any tensors
        # # and then read it in some callback or in `training_epoch_end()` below
        # # remember to always return loss from `training_step()` or backpropagation will fail!
        # return {"loss": loss}
        return {"loss": loss}

    def on_validation_start(self):
        self.val_text_features = OrderedDict()
        self.val_metadata = OrderedDict()

    def validation_step(self, batch: Any, batch_idx: int):
        if self.hparams.val_mode == "text":
            return self.model_step_with_image_text(batch)
        elif self.hparams.val_mode == "label":
            # metadata
            assert isinstance(batch["metadata"], pd.DataFrame), "metadata must be dataframe"
            self.val_metadata[batch_idx] = batch["metadata"]
            # prompt
            if len(self.val_text_features) == 0:
                for (
                    concept_col,
                    caption_str_tokenized_dict,
                ) in self.trainer.datamodule.static_data.items():
                    token_ensemble = []
                    for _, (
                        _,
                        caption_tokenized,
                    ) in caption_str_tokenized_dict.items():
                        token_ensemble.append(caption_tokenized)

                    token_ensemble = torch.concatenate(token_ensemble, dim=0)
                    text_features = self.model_step_with_text(
                        {"text": token_ensemble.to(batch["image"].device)}
                    )["text_features"]
                    self.val_text_features[concept_col] = text_features.detach().cpu()
            # image
            return self.model_step_with_image(batch)
        else:
            raise ValueError("Invalid train mode")

    # def on_validation_batch_end(trainer, outputs, batch, batch_idx, dataloader_idx):
    #     print("on_validation_batch_end", batch_idx, batch["metadata"])
    #     # self.val_metadata.append((batch_idx, batch["metadata"]))
    #     return

    def validation_step_end(self, batch_parts):
        if self.hparams.val_mode == "text":
            logits = self.features_to_logits(
                image_features=batch_parts["image_features"],
                text_features=batch_parts["text_features"],
            )
            loss = self.logits_to_loss(
                logits_per_image=logits["logits_per_image"],
                logits_per_text=logits["logits_per_text"],
            )
            # update and log metrics
            self.val_loss(loss)
            self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": loss}
        elif self.hparams.val_mode == "label":
            return {
                "image_features": batch_parts["image_features"].detach().cpu(),
            }

    def validation_epoch_end(self, outputs: List[Any]):
        if self.hparams.val_mode == "text":
            loss = self.val_loss.compute()  # get current val acc
            self.val_loss_best(loss)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
            # self.log("val/loss_epoch", loss, prog_bar=True)
        elif self.hparams.val_mode == "label":
            # image features
            image_features = torch.concat([out["image_features"] for out in outputs], axis=0)
            # metadata
            metadata_all = pd.concat(
                [self.val_metadata[key] for key in sorted(self.val_metadata.keys())]
            )

            # text features
            text_features_dict = self.val_text_features

            auc_dict = skincon_calcualte_auc_all(
                image_features=image_features,
                text_features_dict=text_features_dict,
                metadata_all=metadata_all,
            )

            auc_all = []
            for concept_col, auc in auc_dict.items():
                self.log(
                    f"val/auc/{concept_col.replace('/','-')}",
                    auc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                if np.isnan(auc):
                    print(f"auc is nan for {concept_col}")
                else:
                    auc_all.append(auc)
            auc_all = np.array(auc_all).mean()
            self.log(
                "val/auc",
                auc_all,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.val_auc_best(auc_all)
            self.log("val/auc_best", self.val_auc_best.compute(), prog_bar=True)

        else:
            raise ValueError("Invalid train mode")

    def on_validation_end(self):
        self.val_text_features = OrderedDict()
        self.val_metadata = OrderedDict()

    def on_test_start(self):
        self.test_text_features = OrderedDict()
        self.test_metadata = OrderedDict()

    def test_step(self, batch: Any, batch_idx: int):
        if self.hparams.test_mode == "text":
            return self.model_step_with_image_text(batch)
        elif self.hparams.test_mode == "label":
            assert isinstance(batch["metadata"], pd.DataFrame), "metadata must be dict"
            # assert batch["static_data"] is not None, "static_data must be not None"
            # metadata
            self.test_metadata[batch_idx] = batch["metadata"]
            # prompt
            # prompt
            if len(self.test_text_features) == 0:
                for (
                    concept_col,
                    caption_str_tokenized_dict,
                ) in self.trainer.static_data.items():
                    caption_tokenized_all = []
                    for _, (
                        _,
                        caption_tokenized,
                    ) in caption_str_tokenized_dict.items():
                        caption_tokenized_all.append(caption_tokenized)

                    caption_tokenized_all = torch.concatenate(caption_tokenized_all, dim=0)
                    text_features = self.model_step_with_text(
                        {"text": caption_tokenized_all.to(batch["image"].device)}
                    )["text_features"]
                    self.test_text_features[concept_col] = text_features.detach().cpu()
            # image
            return self.model_step_with_image(batch)
        else:
            raise ValueError("Invalid train mode")

    def test_step_end(self, batch_parts):
        if self.hparams.test_mode == "text":
            logits = self.features_to_logits(
                image_features=batch_parts["image_features"],
                text_features=batch_parts["text_features"],
            )
            loss = self.logits_to_loss(
                logits_per_image=logits["logits_per_image"],
                logits_per_text=logits["logits_per_text"],
            )
            # update and log metrics
            self.test_loss(loss)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

            return {"loss": loss}
        elif self.hparams.test_mode == "label":
            # print("vali#dation_step_end", batch_parts["metadata"])
            return {
                "image_features": batch_parts["image_features"].detach().cpu(),
            }

    def test_epoch_end(self, outputs: List[Any]):
        if self.hparams.test_mode == "text":
            pass
            # loss = self.test_loss.compute()  # get current test acc
            # self.test_loss_best(loss)  # update best so far test acc
            # log `test_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            # self.log("test/loss_best", self.test_loss_best.compute(), prog_bar=True)
            # self.log("test/loss_epoch", loss, prog_bar=True)
        elif self.hparams.test_mode == "label":
            # image features
            image_features = torch.concat([out["image_features"] for out in outputs], axis=0)
            # metadata
            metadata_all = pd.concat(
                [self.test_metadata[key] for key in sorted(self.test_metadata.keys())]
            )

            # text features
            text_features_dict = self.test_text_features

            auc_dict = skincon_calcualte_auc_all(
                image_features=image_features,
                text_features_dict=text_features_dict,
                metadata_all=metadata_all,
            )

            auc_all = []
            for concept_col, auc in auc_dict.items():
                self.log(
                    f"test/auc/{concept_col.replace('/','-')}",
                    auc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                if np.isnan(auc):
                    print(f"auc is nan for {concept_col}")
                else:
                    auc_all.append(auc)
            auc_all = np.array(auc_all).mean()
            self.log(
                "test/auc",
                auc_all,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            # self.val_target_best(-auc_all)
        else:
            raise ValueError("Invalid train mode")

        pass

    def on_test_end(self):
        self.test_text_features = OrderedDict()
        self.test_metadata = OrderedDict()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


# def play():
#     pass

# TODO: need to confirm whether the parameters are correct. eps is 1.0e-6?
# TODO: number of warm up steps is 10000?
# TODO: unwrap_model(model).logit_scale.clamp_(0, math.log(100))? (original paper)
# TODO: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0) ? (not default)
# TODO: gradient accumulation (not default)

# # First, cache the features without any gradient tracking.
# with torch.no_grad():
#     with autocast():
#         chunk_image_features, chunk_text_features, _ = model(images, texts)
#     accum_image_features.append(chunk_image_features)
#     accum_text_features.append(chunk_text_features)

#     accum_images.append(images)
#     accum_texts.append(texts)

# # If (i + 1) % accum_freq is not zero, move on to the next batch.
# if ((i + 1) % args.accum_freq) > 0:
#     # FIXME this makes data time logging unreliable when accumulating
#     continue

# # Now, ready to take gradients for the last accum_freq batches.
# # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
# # Call backwards each time, but only step optimizer at the end.
# optimizer.zero_grad()
# for j in range(args.accum_freq):
#     images = accum_images[j]
#     texts = accum_texts[j]
#     with autocast():
#         chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
#         image_features = torch.cat(accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
#         text_features = torch.cat(accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
#         total_loss = loss(image_features, text_features, logit_scale)
#     backward(total_loss, scaler)

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "contrastive.yaml")
    _ = hydra.utils.instantiate(cfg)
    print(_)
    _.data
