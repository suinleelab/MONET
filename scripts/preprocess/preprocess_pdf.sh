#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#######################################
# PDF files
#######################################

# 1. prepare files
# 1.1 extract images from pdf files and save them to a folder
python src/MONET/preprocess/pdf_extract.py \
--input data/textbook/pdf_files \
--output data/textbook/pdf_extracted \
--thread 4

# 2. quality control
# 2.1 globbing files
python src/MONET/preprocess/glob_files.py \
--input data/textbook/pdf_extracted \
--output data/textbook/pdf_extracted.compact.hdf5 \
--field images \
--binary \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png

# test
# python src/MONET/preprocess/build_hdf5.py \
# --input data/textbook/pdf_extracted/soyer2012 \
# --output data/textbook/pdf_extracted_temp \
# --style slash_to_underscore \
# --extension jpg jpeg png

# image sanity check

taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input data/textbook/pdf_extracted.compact.hdf5 \
--output data/textbook/pdf_extracted.compact.uncorrupted.hdf5 \
--field images

# featurize images
python src/MONET/preprocess/featurize.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.hdf5 \
--output data/textbook/pdf_extracted.compact.uncorrupted.featurized.pt \
--device cuda:3

# cluster images
python src/MONET/preprocess/cluster.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.hdf5 \
--featurized-file data/textbook/pdf_extracted.compact.uncorrupted.featurized.pt \
--output data/textbook/pdf_extracted.compact.uncorrupted.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

# filter images
python src/MONET/preprocess/filter.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.hdf5 \
--label-file data/textbook/pdf_extracted.compact.uncorrupted.clustering.efficientnet.pca/kmeans_label_lower.csv \
--exclude-label 00 03 07 14 15 18 19 \
09_00 09_05 09_11 09_12 09_14 09_16 09_19 \
16_18 \
--output data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv1.hdf5


# cluster images
python src/MONET/preprocess/cluster.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv1.hdf5 \
--featurized-file data/textbook/pdf_extracted.compact.uncorrupted.featurized.pt \
--output data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv1.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

# filter images
python src/MONET/preprocess/filter.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv1.hdf5 \
--label-file data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv1.clustering.efficientnet.pca/kmeans_label_lower.csv \
--exclude-label \
02_04 02_06 02_15 \
10_00 10_02 10_03 10_04 10_09 10_15 10_17 \
--output data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv2.hdf5

python src/MONET/preprocess/cluster.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv2.hdf5 \
--featurized-file data/textbook/pdf_extracted.compact.uncorrupted.featurized.pt \
--output data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv2.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

python src/MONET/preprocess/filter.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv2.hdf5 \
--label-file data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv2.clustering.efficientnet.pca/kmeans_label_lower.csv \
--exclude-label \
14_06 14_10 14_11 14_12 \
--output data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv3.hdf5

python src/MONET/preprocess/cluster.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv3.hdf5 \
--featurized-file data/textbook/pdf_extracted.compact.uncorrupted.featurized.pt \
--output data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv3.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

# match image and text
python src/MONET/preprocess/pdf_match.py \
--image data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv3.hdf5 \
--pdf-extracted data/textbook/pdf_extracted \
--config data/textbook/pdf_files.config.json \
--output data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv3.matched.pkl

# copy
cp data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv3.matched.csv data/textbook/final_text.csv;
python src/MONET/preprocess/save_as_path.py \
--input data/textbook/pdf_extracted.compact.uncorrupted.dermonlyv3.hdf5 \
--field images \
--output data/textbook/final_image

# syncing
cp -r data/textbook/final* /sdata/chanwkim/dermatology_datasets/textbook
scp chanwkim
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/textbook/final* /sdata/chanwkim/dermatology_datasets/textbook/
scp -r chanwkim@l3:/sdata/chanwkim/dermatology_datasets/pubmed/final* /sdata/chanwkim/dermatology_datasets/pubmed/

######################### deprecated #########################

python src/train.py logger=wandb trainer=gpu trainer.devices=[2] logger.wandb.name="test" seed=42
