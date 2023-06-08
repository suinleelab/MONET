#!/bin/bash

python src/MONET/preprocess/glob_files.py \
--input data/fitzpatrick17k/images \
--output data/fitzpatrick17k/images.compact.pkl \
--field images \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input data/fitzpatrick17k/images.compact.pkl \
--output data/fitzpatrick17k/images.compact.uncorrupted.pkl \
--field images

python src/MONET/preprocess/featurize.py \
--input data/fitzpatrick17k/images.compact.uncorrupted.pkl \
--output data/fitzpatrick17k/images.compact.uncorrupted.featurized.pt \
--device cuda:5;

python src/MONET/preprocess/cluster.py \
--input data/fitzpatrick17k/images.compact.uncorrupted.pkl \
--featurized-file data/fitzpatrick17k/images.compact.uncorrupted.featurized.pt \
--output data/fitzpatrick17k/images.compact.uncorrupted.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

06_00 06_06 15_17

# import pandas as pd

# metadata_all=pd.read_csv("data/fitzpatrick17k/fitzpatrick17k.csv")
# metadata_all["ImageID"]=metadata_all["md5hash"].apply(lambda x: x+'.jpg')

# skincon=pd.read_csv("data/fitzpatrick17k/annotations_fitzpatrick17k.csv")
# assert pd.Index(skincon["ImageID"]).isin(metadata_all["ImageID"]).all(), "Some images are not in the metadata"

# skincon.columns=skincon.columns.map(lambda x: "skincon_"+x)
# metadata_all=metadata_all.merge(skincon, left_on="ImageID", right_on="skincon_ImageID", how="left")
# metadata_all=metadata_all.drop(columns=["skincon_ImageID"])
# metadata_all.to_pickle("data/fitzpatrick17k/fitzpatrick17k.metadata.pkl")

cp data/fitzpatrick17k/fitzpatrick17k.metadata.pkl /sdata/chanwkim/dermatology_datasets/fitzpatrick17k/final_metadata_all.pkl
python src/MONET/preprocess/save_as_path.py \
--input data/fitzpatrick17k/images.compact.uncorrupted.pkl \
--field images \
--output /sdata/chanwkim/dermatology_datasets/fitzpatrick17k/final_image
