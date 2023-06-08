#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#######################################
# PubMed OA
#######################################

# 1. prepare files
# 1.1 search term
python src/MONET/preprocess/pubmed_search.py \
--query-file data/pubmed/search_query.csv \
--start-year 1990 \
--end-year 2023 \
--output data/pubmed/search_csv \
--thread 8

# 1.2 filter with OA dataset
python src/MONET/preprocess/pubmed_download.py filter \
--input data/pubmed/search_csv/all.csv \
--output data/pubmed/oa_file_list.csv

# 1.3 download dataset
python src/MONET/preprocess/pubmed_download.py download \
--input data/pubmed/oa_file_list.csv \
--output /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp .xml,.nxml \
--thread 32
rsync -r -v -e ssh chanwkim@l0:/sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov.3/* /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov
rsync -r -v -e ssh chanwkim@l1lambda:/data2/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov.1/* /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov
rsync -r -v -e ssh chanwkim@l2lambda:/data2/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov.2/* /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov
rsync -r -v -e ssh chanwkim@l2lambda:/data2/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov.4/* /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov
rsync -r -v /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov.0/* /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov

# 2. quality control
# 2.1 globbing files
python src/MONET/preprocess/glob_files.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.pkl \
--field images \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

# 2.2 image sanity check
taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.pkl \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.pkl \
--field images

# featurize images
############## DIVIDE ##############
python src/MONET/preprocess/divide.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.pkl \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.pkl \
--field images \
--num 10

for i in {0..9}
do
    python src/MONET/preprocess/featurize.py \
    --input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided."$i".pkl \
    --output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided."$i".featurized.pt \
    --device cuda:3;
done

python src/MONET/preprocess/merge_files.py \
--input \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.0.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.1.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.2.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.3.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.4.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.5.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.6.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.7.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.8.featurized.pt \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.divided.9.featurized.pt \
--field efficientnet_feature \
--output \
/sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.unordered.featurized.pt
############## DIVIDE ##############

# cluster images
python src/MONET/preprocess/cluster.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.pkl \
--featurized-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.unordered.featurized.pt \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

python src/MONET/preprocess/filter.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.pkl \
--label-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.clustering.efficientnet.pca/kmeans_label_lower.csv \
--exclude-label 00 01 02 05 06 07 08 09 10 11 12 14 16 17 18 \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv1.pkl


python src/MONET/preprocess/cluster.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv1.pkl \
--featurized-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.unordered.featurized.pt \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv1.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

python src/MONET/preprocess/filter.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv1.pkl \
--label-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv1.clustering.efficientnet.pca/kmeans_label_lower.csv \
--exclude-label 00 01 08 10 11 12 13 14 17 18 \
02_00 02_01 02_02 02_03 02_04 02_05 02_06 02_07 02_09 02_12 02_13 02_14 02_15 02_16 02_17 02_18 02_19 \
03_00 03_03 03_04 03_06 03_07 03_08 03_09 03_14 03_15 03_17 03_18 03_19 \
04_05 04_16 04_18 \
05_00 05_01 05_03 05_05 05_06 05_08 05_10 05_11 05_12 05_13 05_14 05_15 05_16 05_17 05_18 05_19 \
06_00 06_01 06_02 06_03 06_04 06_05 06_06 06_07 06_08 06_09 06_10 06_11 06_14 06_16 06_17 06_18 06_19 \
07_00 07_01 07_02 07_04 07_05 07_06 07_07 07_08 07_09 07_10 07_11 07_12 07_13 07_14 07_15 07_16 07_17 07_18 07_19 \
15_00 15_01 15_02 15_03 15_04 15_05 15_07 15_08 15_09 15_10 15_11 15_12 15_13 15_14 15_15 15_16 15_17 15_18 15_19 \
16_01 16_02 16_03 16_04 16_05 16_07 16_09 16_10 16_11 16_12 16_13 16_14 16_15 16_16 16_17 16_18 16_19 \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv2.pkl
# " ".join([f"16_{i:02d}" for i in range(20)])


python src/MONET/preprocess/cluster.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv2.pkl \
--featurized-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.unordered.featurized.pt \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv2.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20


python src/MONET/preprocess/filter.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv2.pkl \
--label-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv2.clustering.efficientnet.pca/kmeans_label_lower.csv \
--exclude-label 17 \
00_10 00_12 00_13 00_14 00_15 00_17 00_19 \
02_00 02_01 \
03_00 03_01 03_02 03_03 03_04 03_05 03_06 03_07 03_09 03_10 03_12 03_13 03_14 03_15 03_16 03_18 03_19 \
05_00 05_01 05_02 05_05 05_06 05_07 05_09 05_10 05_11 05_12 05_15 05_17 05_18 \
06_00 06_01 06_03 06_04 06_09 06_11 06_18 06_19 \
07_00 07_01 07_02 07_04 07_05 07_06 07_07 07_08 07_09 07_10 07_11 07_14 07_15 07_18 07_19 \
09_00 09_03 09_05 09_08 09_09 09_10 09_12 09_13 09_14 09_15 09_16 09_17 \
10_00 10_01 10_02 10_03 10_06 10_07 10_08 10_09 10_10 10_11 10_12 10_13 10_14 10_16 10_17 10_18 10_19 \
11_00 11_02 11_04 11_05 11_09 11_13 11_14 11_15 11_16 11_17 \
14_00 14_01 14_02 14_03 14_04 14_05 14_06 14_07 14_09 14_10 14_11 14_12 14_13 14_15 14_16 14_17 14_18 \
18_12 \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv3.pkl


python src/MONET/preprocess/cluster.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv3.pkl \
--featurized-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.unordered.featurized.pt \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv3.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

python src/MONET/preprocess/filter.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv3.pkl \
--label-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv3.clustering.efficientnet.pca/kmeans_label_lower.csv \
--exclude-label \
04_00 04_01 04_04 04_05 04_06 04_07 04_08 04_09 04_10 04_11 04_12 04_13 04_14 04_15 \
09_01 09_02 09_06 09_12 09_13 09_15 09_16 09_17 09_18 09_19 \
13_00 13_01 13_03 13_04 13_05 13_06 13_07 13_08 13_09 13_10 13_11 13_13 13_14 13_15 13_16 13_17 13_18 13_19 \
14_04 14_13 14_14 \
16_12 \
17_07 17_08 17_09 17_10 \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.pkl


python src/MONET/preprocess/cluster.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.pkl \
--featurized-file /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.unordered.featurized.pt \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.clustering.efficientnet.pca \
--pca \
--feature-to-use efficientnet \
-n1 20 -n2 20

# globbing xml files
python src/MONET/preprocess/glob_files.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/ftp.ncbi.nlm.nih.gov \
--output /sdata/chanwkim/dermatology_datasets/pubmed/xml.pkl \
--field xml \
--style slash_to_underscore \
--extension .xml,.nxml;


# image sanity check again
taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.pkl \
--output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.doublecheck.pkl \
--field images \
--relative-path

# match image and text
python src/MONET/preprocess/pubmed_match.py \
--image /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.pkl \
--xml /sdata/chanwkim/dermatology_datasets/pubmed/xml.pkl \
--output /sdata/chanwkim/dermatology_datasets/pubmed/xml.matched.pkl


# taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
# --input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.pkl \
# --output /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.doublecheck_.pkl \
# --field images \
# --relative-path

# finalize
cp /sdata/chanwkim/dermatology_datasets/pubmed/xml.matched.pkl /sdata/chanwkim/dermatology_datasets/pubmed/final_text.pkl;

python src/MONET/preprocess/save_as_path.py \
--input /sdata/chanwkim/dermatology_datasets/pubmed/images.uncorrupted.dermonlyv4.pkl \
--field images \
--output /sdata/chanwkim/dermatology_datasets/pubmed/final_image


# taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
# --input /sdata/chanwkim/dermatology_datasets/pubmed/final_image.pkl \
# --output /sdata/chanwkim/dermatology_datasets/pubmed/final_image.uncorrupted.pkl \
# --field images \
# --relative-path
