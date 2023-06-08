#!/bin/bash
python src/MONET/preprocess/glob_files.py \
--input data/ddi/ddidiversedermatologyimages \
--output data/ddi/ddidiversedermatologyimages.compact.pkl \
--field images \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input data/ddi/ddidiversedermatologyimages.compact.pkl \
--output data/ddi/ddidiversedermatologyimages.compact.uncorrupted.pkl \
--field images

python src/MONET/preprocess/featurize.py \
--input  data/ddi/ddidiversedermatologyimages.compact.uncorrupted.pkl \
--output  data/ddi/ddidiversedermatologyimages.compact.uncorrupted.featurized.pkl \
--device cuda:5;

python src/MONET/preprocess/cluster.py \
--input data/ddi/ddidiversedermatologyimages.compact.uncorrupted.pkl \
--featurized-file data/ddi/ddidiversedermatologyimages.compact.uncorrupted.featurized.pkl \
--output data/ddi/ddidiversedermatologyimages.compact.uncorrupted.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

# 06_00 06_06 15_17
# data/ddi/ddidiversedermatologyimages.compact.uncorrupted.clustering.efficientnet.pca

# /projects/leelab2/chanwkim/dermatology_datasets/ddi/ddidiversedermatologyimages/

# import pandas as pd

# metadata_all=pd.read_csv("data/ddi/ddidiversedermatologyimages/ddi_metadata.csv")
# metadata_all["ImageID"]=metadata_all["md5hash"].apply(lambda x: x+'.jpg')

# skincon=pd.read_csv("data/ddi/annotations_ddi.csv")
# assert pd.Index(skincon["ImageID"]).isin(metadata_all["DDI_file"]).all(), "Some images are not in the metadata"

# skincon.columns=skincon.columns.map(lambda x: "skincon_"+x)
# metadata_all=metadata_all.merge(skincon, left_on="DDI_file", right_on="skincon_ImageID", how="left")
# metadata_all=metadata_all.drop(columns=["skincon_ImageID"])
# metadata_all.to_pickle("data/ddi/ddi.metadata.pkl")

cp data/ddi/ddi.metadata.pkl /sdata/chanwkim/dermatology_datasets/ddi/final_metadata_all.pkl
python src/MONET/preprocess/save_as_path.py \
--input data/ddi/ddidiversedermatologyimages.compact.uncorrupted.pkl \
--field images \
--output /sdata/chanwkim/dermatology_datasets/ddi/final_image
