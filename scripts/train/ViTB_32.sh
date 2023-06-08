#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# cp data/textbook_final/text.csv /sdata/chanwkim/dermatology_datasets/textbook/text_final.csv
# cp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /sdata/chanwkim/dermatology_datasets/pubmed/

# cp data/textbook_final/text.csv /data2/chanwkim/dermatology_datasets/textbook/text_final.csv
# cp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /data2/chanwkim/dermatology_datasets/pubmed/

#cp data/fitzpatrick17k/images.csv /data2/chanwkim/dermatology_datasets/textbook/text_final.csv

# scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /data2/chanwkim/dermatology_datasets/pubmed/
tar -cvf /sdata/chanwkim/dermatology_datasets.tar /sdata/chanwkim/dermatology_datasets/

scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /data2/chanwkim/dermatology_datasets/pubmed/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/textbook/final* /data2/chanwkim/dermatology_datasets/textbook/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/fitzpatrick17k/final* /data2/chanwkim/dermatology_datasets/fitzpatrick17k/

scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /sdata/chanwkim/dermatology_datasets/pubmed/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/textbook/final* /sdata/chanwkim/dermatology_datasets/textbook/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/fitzpatrick17k/final* /sdata/chanwkim/dermatology_datasets/fitzpatrick17k/
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/ddi/final* /sdata/chanwkim/dermatology_datasets/ddi/;

mkdir /scr/chanwkim/dermatology_datasets
mkdir /scr/chanwkim/dermatology_datasets/pubmed;
mkdir /scr/chanwkim/dermatology_datasets/textbook;
mkdir /scr/chanwkim/dermatology_datasets/fitzpatrick17k;
mkdir /scr/chanwkim/dermatology_datasets/ddi;

scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /scr/chanwkim/dermatology_datasets/pubmed/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/textbook/final* /scr/chanwkim/dermatology_datasets/textbook/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/fitzpatrick17k/final* /scr/chanwkim/dermatology_datasets/fitzpatrick17k/
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/ddi/final* /scr/chanwkim/dermatology_datasets/ddi/;





mkdir /scr/chanwkim/dermatology_datasets/;
mkdir /scr/chanwkim/dermatology_datasets/pubmed/;
mkdir /scr/chanwkim/dermatology_datasets/textbook/;
mkdir /scr/chanwkim/dermatology_datasets/fitzpatrick17k/;
mkdir /scr/chanwkim/dermatology_datasets/ddi/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /scr/chanwkim/dermatology_datasets/pubmed/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/textbook/final* /scr/chanwkim/dermatology_datasets/textbook/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/fitzpatrick17k/final* /scr/chanwkim/dermatology_datasets/fitzpatrick17k/;
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/ddi/final* /scr/chanwkim/dermatology_datasets/ddi/;
#scp -r chanwkim@l0:/sdata/chanwkim/dermatology_datasets.tar /scr/chanwkim/dermatology_datasets/




python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_allpubmedtextbook_512_evalboth" seed=42 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
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

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_allpubmedtextbook_512_evalboth" seed=43 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_all,textbook_all\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val,textbook_val\' \
datamodule.dataset_name_test=\'pubmed_test,textbook_test\' \
datamodule.random_state=43 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"




#######

# ViT-B/32

python src/eval.py \
logger=wandb logger.wandb.name="ViT-B/32_original_fitzpatrick17k" paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'fitzpatrick17k_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
ckpt_path=null

python src/eval.py \
logger=wandb logger.wandb.name="ViT-B/32_original_ddi" paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'ddi_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
ckpt_path=null

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_128" seed=42 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'pubmed_val\' \
datamodule.dataset_name_test=\'pubmed_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_128_1e-4" seed=42 paths=klone \
trainer=dp trainer.devices=[0,1,2,3] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'pubmed_val\' \
datamodule.dataset_name_test=\'pubmed_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min" \
model.optimizer.lr=1e-4

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512_1e-6" seed=42 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val\' \
datamodule.dataset_name_test=\'pubmed_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min" \
model.optimizer.lr=1e-6

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512_2e-5" seed=42 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val\' \
datamodule.dataset_name_test=\'pubmed_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min" \
model.optimizer.lr=2e-5

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512_5e-5" seed=42 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val\' \
datamodule.dataset_name_test=\'pubmed_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min" \
model.optimizer.lr=5e-5

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmedtextbook_512_evalboth" seed=42 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train,textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val,textbook_val\' \
datamodule.dataset_name_test=\'pubmed_test,textbook_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512_evalboth" seed=42 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val,textbook_val\' \
datamodule.dataset_name_test=\'pubmed_test,textbook_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_textbook_512_evalboth" seed=42 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val,textbook_val\' \
datamodule.dataset_name_test=\'pubmed_test,textbook_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"




python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512" seed=42 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val\' \
datamodule.dataset_name_test=\'pubmed_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:4 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512_1e-4" seed=42 paths=klone \
trainer=dp trainer.devices=[0,1,2,3] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'pubmed_val\' \
datamodule.dataset_name_test=\'pubmed_test\' \
datamodule.random_state=42 \
model.train_mode="text" \
model.val_mode="text" \
model.test_mode="text" \
model.net.device=cuda:0 \
callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min" \
model.optimizer.lr=1e-4

################################################
python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512_seed42" seed=42 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmed_512_seed45" seed=45 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"



python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_textbook_128" seed=42 paths=l2lambda \
trainer=dp trainer.devices=[0,1,2,3] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'textbook_train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'pubmed_test\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_textbook_512" seed=42 paths=l2lambda \
trainer=dp trainer.devices=[0,1,2,3] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'pubmed_test\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_textbook_512_seed45" seed=42 paths=l2lambda \
trainer=dp trainer.devices=[0,1,2,3] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"


python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmedtextbook_128_test" seed=42 paths=l3 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train,textbook_train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmedtextbook_128" seed=42 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train,textbook_train\'  datamodule.batch_size_train=128 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmedtextbook_512" seed=42 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train,textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmedtextbook_512_seed45" seed=45 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train,textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"


python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmedtextbook_512_seed45" seed=45 paths=l0 \
trainer=dp trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train,textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max"

python src/train.py \
logger=wandb logger.wandb.name="ViT-B/32_pubmedtextbook_512_seed45" seed=45 paths=l0 \
trainer=fsdp_native trainer.devices=[4,5,6,7] \
model.net.model_name_or_path=ViT-B/32 \
datamodule=multiplex \
datamodule.num_workers=8 datamodule.pin_memory=True \
datamodule.dataset_name_train=\'pubmed_train,textbook_train\'  datamodule.batch_size_train=512 \
datamodule.dataset_name_val=\'fitzpatrick17k_all\' \
datamodule.dataset_name_test=\'ddi_all\' \
model.train_mode="text" \
model.val_mode="label" \
model.test_mode="label" \
callbacks.model_checkpoint.monitor="val/auc" callbacks.model_checkpoint.mode="max" \
+trainer.precision=16
