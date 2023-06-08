import torch
from torch.nn import functional as F
from torchmetrics import (
    AUROC,
    Accuracy,
    CohenKappa,
    F1Score,
    MeanMetric,
    Precision,
    Recall,
)
from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import AdamW


def set_schedule(pl_module):
    optimizer = None
    if pl_module.hparams.optim_type is None:
        return (
            [None],
            [None],
        )
    else:
        if pl_module.hparams.optim_type == "Adamw":
            optimizer = AdamW(
                params=pl_module.parameters(),
                lr=pl_module.hparams.learning_rate,
                weight_decay=pl_module.hparams.weight_decay,
            )
        elif pl_module.hparams.optim_type == "Adam":
            optimizer = torch.optim.Adam(
                pl_module.parameters(),
                lr=pl_module.hparams.learning_rate,
                weight_decay=pl_module.hparams.weight_decay,
            )
        elif pl_module.hparams.optim_type == "SGD":
            optimizer = torch.optim.SGD(
                pl_module.parameters(),
                lr=pl_module.hparams.learning_rate,
                momentum=0.9,
                weight_decay=pl_module.hparams.weight_decay,
            )

        if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
            max_steps = (
                len(pl_module.trainer.datamodule.train_dataloader())
                * pl_module.trainer.max_epochs
                // pl_module.trainer.accumulate_grad_batches
            )
        else:
            max_steps = pl_module.trainer.max_steps

        if pl_module.hparams.decay_power == "cosine":
            scheduler = {
                "scheduler": get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=pl_module.hparams.warmup_steps,
                    num_training_steps=max_steps,
                ),
                "interval": "step",
            }
        else:
            raise NotImplementedError("Only cosine scheduler is implemented for now")

        return (
            [optimizer],
            [scheduler],
        )


def set_metrics(pl_module, num_labels=None):
    for phase in ["train", "val", "test"]:
        if pl_module.hparams.target_type == "binary":
            setattr(pl_module, f"{phase}_loss", MeanMetric())

            setattr(pl_module, f"{phase}_accuracy", Accuracy(task="binary"))
            setattr(
                pl_module,
                f"{phase}_precision",
                Precision(
                    task="binary",
                    num_classes=pl_module.hparams.output_dim,
                    average="macro",
                ),
            )
            setattr(
                pl_module,
                f"{phase}_recall",
                Recall(
                    task="binary",
                    num_classes=pl_module.hparams.output_dim,
                    average="macro",
                ),
            )
            setattr(
                pl_module,
                f"{phase}_f1",
                F1Score(
                    task="binary",
                    num_classes=pl_module.hparams.output_dim,
                    average="macro",
                ),
            )
            setattr(
                pl_module,
                f"{phase}_cohenkappa",
                CohenKappa(task="binary", weights="quadratic"),
            )
            setattr(pl_module, f"{phase}_auroc", AUROC(task="binary"))

        elif pl_module.hparams.target_type == "multiclass":
            setattr(pl_module, f"{phase}_loss", MeanMetric())

            setattr(
                pl_module,
                f"{phase}_accuracy",
                Accuracy(task="multiclass", num_classes=pl_module.hparams.output_dim),
            )
            setattr(
                pl_module,
                f"{phase}_precision",
                Precision(
                    task="multiclass",
                    num_classes=pl_module.hparams.output_dim,
                    average="macro",
                ),
            )
            setattr(
                pl_module,
                f"{phase}_recall",
                Recall(
                    task="multiclass",
                    num_classes=pl_module.hparams.output_dim,
                    average="macro",
                ),
            )
            setattr(
                pl_module,
                f"{phase}_f1",
                F1Score(
                    task="multiclass",
                    num_classes=pl_module.hparams.output_dim,
                    average="macro",
                ),
            )
            setattr(
                pl_module,
                f"{phase}_cohenkappa",
                CohenKappa(
                    task="multiclass",
                    num_classes=pl_module.hparams.output_dim,
                    weights="quadratic",
                ),
            )

        elif "multilabel" in pl_module.hparams.target_type:
            output_select = pl_module.hparams.target_type.split("-")[1].replace("loss", "")
            if output_select == "all":
                setattr(pl_module, f"{phase}_loss", MeanMetric())
            else:
                setattr(pl_module, f"{phase}_loss", MeanMetric())

            output_select = pl_module.hparams.target_type.split("-")[2].replace("metric", "")
            if output_select == "all":
                setattr(
                    pl_module,
                    f"{phase}_auroc",
                    AUROC(task="multilabel", num_labels=num_labels),
                )
            else:
                setattr(pl_module, f"{phase}_accuracy", Accuracy(task="binary"))
                setattr(
                    pl_module,
                    f"{phase}_precision",
                    Precision(num_classes=1, task="binary", average="macro"),
                )
                setattr(
                    pl_module,
                    f"{phase}_recall",
                    Recall(num_classes=1, task="binary", average="macro"),
                )
                setattr(
                    pl_module,
                    f"{phase}_f1",
                    F1Score(num_classes=1, task="binary", average="macro"),
                )
                setattr(pl_module, f"{phase}_auroc", AUROC(task="binary"))
        elif pl_module.hparams.target_type == "regression":
            setattr(pl_module, f"{phase}_loss", MeanMetric())
        else:
            raise NotImplementedError(
                "Not supported target type. It should be one of binary, multiclass, multilabel"
            )


def epoch_wrapup(pl_module, phase):
    if pl_module.hparams.target_type == "binary":
        loss = getattr(pl_module, f"{phase}_loss").compute()
        getattr(pl_module, f"{phase}_loss").reset()
        pl_module.log(f"{phase}/epoch_loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy").compute()
        getattr(pl_module, f"{phase}_accuracy").reset()
        pl_module.log(f"{phase}/epoch_accuracy", accuracy)

        precision = getattr(pl_module, f"{phase}_precision").compute()
        getattr(pl_module, f"{phase}_precision").reset()
        pl_module.log(f"{phase}/epoch_precision", precision)

        recall = getattr(pl_module, f"{phase}_recall").compute()
        getattr(pl_module, f"{phase}_recall").reset()
        pl_module.log(f"{phase}/epoch_recall", recall)

        f1 = getattr(pl_module, f"{phase}_f1").compute()
        getattr(pl_module, f"{phase}_f1").reset()
        pl_module.log(f"{phase}/epoch_f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa").compute()
        getattr(pl_module, f"{phase}_cohenkappa").reset()
        pl_module.log(f"{phase}/epoch_cohenkappa", cohenkappa)

        auroc = getattr(pl_module, f"{phase}_auroc").compute()
        getattr(pl_module, f"{phase}_auroc").reset()
        pl_module.log(f"{phase}/epoch_auroc", auroc)

    elif pl_module.hparams.target_type == "multiclass":
        loss = getattr(pl_module, f"{phase}_loss").compute()
        getattr(pl_module, f"{phase}_loss").reset()
        pl_module.log(f"{phase}/epoch_loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy").compute()
        getattr(pl_module, f"{phase}_accuracy").reset()
        pl_module.log(f"{phase}/epoch_accuracy", accuracy)

        precision = getattr(pl_module, f"{phase}_precision").compute()
        getattr(pl_module, f"{phase}_precision").reset()
        pl_module.log(f"{phase}/epoch_precision", precision)

        recall = getattr(pl_module, f"{phase}_recall").compute()
        getattr(pl_module, f"{phase}_recall").reset()
        pl_module.log(f"{phase}/epoch_recall", recall)

        f1 = getattr(pl_module, f"{phase}_f1").compute()
        getattr(pl_module, f"{phase}_f1").reset()
        pl_module.log(f"{phase}/epoch_f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa").compute()
        getattr(pl_module, f"{phase}_cohenkappa").reset()
        pl_module.log(f"{phase}/epoch_cohenkappa", cohenkappa)

    elif "multilabel" in pl_module.hparams.target_type:
        output_select = pl_module.hparams.target_type.split("-")[1].replace("loss", "")
        if output_select == "all":
            loss = getattr(pl_module, f"{phase}_loss").compute()
            getattr(pl_module, f"{phase}_loss").reset()
            pl_module.log(f"{phase}/epoch_loss", loss)
        else:
            loss = getattr(pl_module, f"{phase}_loss").compute()
            getattr(pl_module, f"{phase}_loss").reset()
            pl_module.log(f"{phase}/epoch_loss", loss)

        output_select = pl_module.hparams.target_type.split("-")[2].replace("metric", "")
        if output_select == "all":
            auroc = getattr(pl_module, f"{phase}_auroc").compute()
            getattr(pl_module, f"{phase}_auroc").reset()
            pl_module.log(f"{phase}/epoch_auroc", auroc)
        else:
            accuracy = getattr(pl_module, f"{phase}_accuracy").compute()
            getattr(pl_module, f"{phase}_accuracy").reset()
            pl_module.log(f"{phase}/epoch_accuracy", accuracy)

            precision = getattr(pl_module, f"{phase}_precision").compute()
            getattr(pl_module, f"{phase}_precision").reset()
            pl_module.log(f"{phase}/epoch_precision", precision)

            recall = getattr(pl_module, f"{phase}_recall").compute()
            getattr(pl_module, f"{phase}_recall").reset()
            pl_module.log(f"{phase}/epoch_recall", recall)

            f1 = getattr(pl_module, f"{phase}_f1").compute()
            getattr(pl_module, f"{phase}_f1").reset()
            pl_module.log(f"{phase}/epoch_f1", f1)

            auroc = getattr(pl_module, f"{phase}_auroc").compute()
            getattr(pl_module, f"{phase}_auroc").reset()
            pl_module.log(f"{phase}/epoch_auroc", auroc)
    elif pl_module.hparams.target_type == "regression":
        loss = getattr(pl_module, f"{phase}_loss").compute()
        getattr(pl_module, f"{phase}_loss").reset()
        pl_module.log(f"{phase}/epoch_loss", loss)
    else:
        raise NotImplementedError(
            "Not supported target type. It should be one of binary, multiclass, multilabel"
        )


def compute_metrics(pl_module, logits, labels, phase):
    # phase = "train" if pl_module.training else "val"
    if pl_module.hparams.target_type == "binary":
        if pl_module.hparams.loss_weight is None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float()
            )  # internally computes sigmoid
        else:
            loss = (
                pl_module.hparams.loss_weight[0]
                / (pl_module.hparams.loss_weight[0] + pl_module.hparams.loss_weight[1])
                * F.binary_cross_entropy_with_logits(
                    logits,
                    labels.float(),
                    pos_weight=torch.tensor(
                        pl_module.hparams.loss_weight[1] / pl_module.hparams.loss_weight[0]
                    ).float(),
                )
            )
            F.binary_cross_entropy_with_logits(
                logits, labels.float(), weight=torch.tensor(5).float()
            )
        loss = getattr(pl_module, f"{phase}_loss")(loss)
        pl_module.log(f"{phase}/loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/accuracy", accuracy)

        recall = getattr(pl_module, f"{phase}_recall")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/recall", recall)

        precision = getattr(pl_module, f"{phase}_precision")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/precision", precision)

        f1 = getattr(pl_module, f"{phase}_f1")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/cohenkappa", cohenkappa)

        auroc = getattr(pl_module, f"{phase}_auroc")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/auroc", auroc)

    elif pl_module.hparams.target_type == "multiclass":
        # import ipdb

        # ipdb.set_trace()

        loss = F.cross_entropy(input=logits, target=labels)  # internally computes softmax
        loss = getattr(pl_module, f"{phase}_loss")(loss)
        pl_module.log(f"{phase}/loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/accuracy", accuracy)

        recall = getattr(pl_module, f"{phase}_recall")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/recall", recall)

        precision = getattr(pl_module, f"{phase}_precision")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/precision", precision)

        f1 = getattr(pl_module, f"{phase}_f1")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa")(
            torch.softmax(logits, dim=1), labels
        )
        pl_module.log(f"{phase}/cohenkappa", cohenkappa)

    elif "multilabel" in pl_module.hparams.target_type:
        output_select = pl_module.hparams.target_type.split("-")[1].replace("loss", "")
        if output_select == "all":
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.type_as(logits)
            )  # internally computes sigmoid
            loss = getattr(pl_module, f"{phase}_loss")(loss)
            pl_module.log(f"{phase}/loss", loss)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits[:, int(output_select)],
                labels[:, int(output_select)].type_as(logits),
            )  # internally computes sigmoid
            loss = getattr(pl_module, f"{phase}_loss")(loss)
            pl_module.log(f"{phase}/loss", loss)

        output_select = pl_module.hparams.target_type.split("-")[2].replace("metric", "")
        if output_select == "all":
            auroc = getattr(pl_module, f"{phase}_auroc")(
                torch.sigmoid(logits[:, :]),
                labels[:, :],
            )
            pl_module.log(f"{phase}/auroc", auroc)
        else:
            accuracy = getattr(pl_module, f"{phase}_accuracy")(
                torch.sigmoid(logits[:, int(output_select)]),
                labels[:, int(output_select)],
            )
            pl_module.log(f"{phase}/accuracy", accuracy)

            precision = getattr(pl_module, f"{phase}_precision")(
                torch.sigmoid(logits[:, int(output_select)]),
                labels[:, int(output_select)],
            )
            pl_module.log(f"{phase}/precision", precision)

            recall = getattr(pl_module, f"{phase}_recall")(
                torch.sigmoid(logits[:, int(output_select)]),
                labels[:, int(output_select)],
            )
            pl_module.log(f"{phase}/recall", recall)

            f1 = getattr(pl_module, f"{phase}_f1")(
                torch.sigmoid(logits[:, int(output_select)]),
                labels[:, int(output_select)],
            )
            pl_module.log(f"{phase}/f1", f1)

            auroc = getattr(pl_module, f"{phase}_auroc")(
                torch.sigmoid(logits[:, int(output_select)]),
                labels[:, int(output_select)],
            )
            pl_module.log(f"{phase}/auroc", auroc)

    if pl_module.hparams.target_type == "regression":
        # print(logits.shape, labels.shape)
        # print(logits.shape, labels.max(axis=1).values)
        # print(labels.max(axis=0).values)
        loss = F.mse_loss(input=logits, target=labels.float())
        loss = getattr(pl_module, f"{phase}_loss")(loss)
        pl_module.log(f"{phase}/loss", loss)
    else:
        raise NotImplementedError(
            "Not supported target type. It should be one of binary, multiclass, multilabel"
        )

    return loss
