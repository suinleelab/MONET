#!/bin/bash


python src/train.py \
logger=wandb logger.wandb.name="ResNet50_fitzpatrick17k_128_clean_ninelabel_1e-3" paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model=classifier \
model.backbone_type="resnet50" model.target_type="multiclass" model.output_dim=9 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'fitzpatrick17k_clean_ninelabel=train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_clean_ninelabel=val\' \
datamodule.dataset_name_test=\'fitzpatrick17k_clean_ninelabel=test\' \
datamodule.split_seed=42 \
callbacks.model_checkpoint.monitor="val/epoch_accuracy" callbacks.model_checkpoint.mode="max" \
model.learning_rate=0.0001 trainer.max_epochs=50

python src/train.py \
logger=wandb logger.wandb.name="ResNet50_fitzpatrick17k_128_unclean_ninelabel_1e-3" paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model=classifier \
model.backbone_type="resnet50" model.target_type="multiclass" model.output_dim=9 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'fitzpatrick17k_ninelabel=train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_ninelabel=val\' \
datamodule.dataset_name_test=\'fitzpatrick17k_ninelabel=test\' \
datamodule.split_seed=42 \
callbacks.model_checkpoint.monitor="val/epoch_accuracy" callbacks.model_checkpoint.mode="max" \
model.learning_rate=0.0001 trainer.max_epochs=50

python src/train.py \
logger=wandb logger.wandb.name="ResNet50_fitzpatrick17k_128_clean_disease_1e-3" paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model=classifier \
model.backbone_type="resnet50" model.target_type="multilabel-all-all" model.output_dim=114 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'fitzpatrick17k_clean_disease=train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_clean_disease=val\' \
datamodule.dataset_name_test=\'fitzpatrick17k_clean_disease=test\' \
datamodule.split_seed=42 \
callbacks.model_checkpoint.monitor="val/epoch_auroc" callbacks.model_checkpoint.mode="max" \
model.learning_rate=0.0001 trainer.max_epochs=50

python src/train.py \
logger=wandb logger.wandb.name="ResNet50_fitzpatrick17k_128_unclean_disease_1e-3" paths=l0 \
trainer=dp trainer.devices=[0,1,2,3] \
model=classifier \
model.backbone_type="resnet50" model.target_type="multilabel-all-all" model.output_dim=1 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'fitzpatrick17k_disease=train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_disease=val\' \
datamodule.dataset_name_test=\'fitzpatrick17k_disease=test\' \
datamodule.split_seed=42 \
callbacks.model_checkpoint.monitor="val/epoch_auroc" callbacks.model_checkpoint.mode="max" \
model.learning_rate=0.0001 trainer.max_epochs=50



python src/train.py \
logger=wandb logger.wandb.name="ResNet50_fitzpatrick17k_128_clean_threelabel_1e-3" paths=l0 \
trainer=dp trainer.devices=[0,1,2,3] \
model=classifier \
model.backbone_type="resnet50" model.target_type="multiclass" model.output_dim=3 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'fitzpatrick17k_clean_threelabel=train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_clean_threelabel=val\' \
datamodule.dataset_name_test=\'fitzpatrick17k_clean_threelabel=test\' \
datamodule.split_seed=42 \
callbacks.model_checkpoint.monitor="val/epoch_accuracy" callbacks.model_checkpoint.mode="max" \
model.learning_rate=0.0001 trainer.max_epochs=50

python src/train.py \
logger=wandb logger.wandb.name="ResNet50_fitzpatrick17k_128_unclean_threelabel_1e-3" paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model=classifier \
model.backbone_type="resnet50" model.target_type="multiclass" model.output_dim=3 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'fitzpatrick17k_threelabel=train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_threelabel=val\' \
datamodule.dataset_name_test=\'fitzpatrick17k_threelabel=test\' \
datamodule.split_seed=42 \
callbacks.model_checkpoint.monitor="val/epoch_accuracy" callbacks.model_checkpoint.mode="max" \
model.learning_rate=0.0001 trainer.max_epochs=50


#################
python src/train.py \
logger=wandb logger.wandb.name="ResNet50_fitzpatrick17k_128_pseudoconcept_1e-3" paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model=classifier \
model.backbone_type="resnet50" model.target_type="regression" model.output_dim=48 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'fitzpatrick17k_pseudo_skinon=train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_pseudo_skinon=val\' \
datamodule.dataset_name_test=\'fitzpatrick17k_pseudo_skinon=test\' \
datamodule.split_seed=42 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min" \
model.learning_rate=0.0001 trainer.max_epochs=50

# from MONET.datamodules.components.base_dataset import BaseDataset
