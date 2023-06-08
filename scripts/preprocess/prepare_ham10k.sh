#!/bin/bash
kaggle datasets download surajghuwalewala/ham1000-segmentation-and-classification
mkdir data/ham10k
mv ham1000-segmentation-and-classification.zip data/ham10k
cd data/ham10k || exit
unzip ham1000-segmentation-and-classification.zip
rm ham1000-segmentation-and-classification.zip
mkdir images
cp HAM10000_images_part_1/* images
cp HAM10000_images_part_2/* images

python src/MONET/preprocess/glob_files.py \
--input data/ham10k/images \
--output data/ham10k/images.compact.pkl \
--field images \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input data/ham10k/images.compact.pkl \
--output data/ham10k/images.compact.uncorrupted.pkl \
--field images


images.compact.uncorrupted.pkl

cp data/ham10k/HAM10000_metadata.csv /sdata/chanwkim/dermatology_datasets/ham10k/final_metadata_all.csv
python src/MONET/preprocess/save_as_path.py \
--input data/ham10000/images.compact.uncorrupted.pkl \
--field images \
--output /sdata/chanwkim/dermatology_datasets/ham10000/final_image
