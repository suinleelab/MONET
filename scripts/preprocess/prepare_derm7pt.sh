#!/bin/bash

python src/MONET/preprocess/glob_files.py \
--input data/derm7pt/images \
--output data/derm7pt/images.compact.pkl \
--field images \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input data/derm7pt/images.compact.pkl \
--output data/derm7pt/images.compact.uncorrupted.pkl \
--field images


images.compact.uncorrupted.pkl

cp data/derm7pt/meta/meta.csv /sdata/chanwkim/dermatology_datasets/derm7pt/final_metadata_all.csv
python src/MONET/preprocess/save_as_path.py \
--input data/derm7pt/images.compact.uncorrupted.pkl \
--field images \
--output /sdata/chanwkim/dermatology_datasets/derm7pt/final_image
