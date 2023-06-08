#!/bin/bash
python src/train.py \
logger=wandb logger.wandb.name="ViT-B/16_allpubmedtextbook_512_evalboth" seed=42 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/16 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_all,textbook_all\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val,textbook_val\' \
datamodule.dataset_name_test=\'pubmed_test,textbook_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"
