isic collection list

┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ ID  ┃ Name                                                                ┃ Public ┃ Pinned ┃ Locked ┃ DOI             ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 249 │ BCN20000                                                            │ True   │ False  │ False  │ None            │
│ 61  │ Challenge 2016: Test                                                │ True   │ True   │ True   │ None            │
│ 74  │ Challenge 2016: Training                                            │ True   │ True   │ True   │ None            │
│ 69  │ Challenge 2017: Test                                                │ True   │ True   │ True   │ None            │
│ 60  │ Challenge 2017: Training                                            │ True   │ True   │ True   │ None            │
│ 71  │ Challenge 2017: Validation                                          │ True   │ True   │ True   │ None            │
│ 64  │ Challenge 2018: Task 1-2: Test                                      │ True   │ True   │ True   │ None            │
│ 63  │ Challenge 2018: Task 1-2: Training                                  │ True   │ True   │ True   │ None            │
│ 62  │ Challenge 2018: Task 1-2: Validation                                │ True   │ True   │ True   │ None            │
│ 67  │ Challenge 2018: Task 3: Test                                        │ True   │ True   │ True   │ None            │
│ 66  │ Challenge 2018: Task 3: Training                                    │ True   │ True   │ True   │ None            │
│ 73  │ Challenge 2018: Task 3: Validation                                  │ True   │ True   │ True   │ None            │
│ 65  │ Challenge 2019: Training                                            │ True   │ True   │ True   │ None            │
│ 70  │ Challenge 2020: Training                                            │ True   │ True   │ True   │ None            │
│ 97  │ Collection for ISBI 2016: 100 Lesion Classification                 │ True   │ False  │ True   │ None            │
│ 216 │ Consecutive biopsies for melanoma across year 2020                  │ True   │ False  │ True   │ 10.34970/151324 │
│ 75  │ Consumer AI apps                                                    │ True   │ False  │ True   │ 10.34970/401946 │
│ 166 │ EASY Dermoscopy Expert Agreement Study                              │ True   │ False  │ False  │ None            │
│ 212 │ HAM10000                                                            │ True   │ True   │ True   │ None            │
│ 175 │ HIBA Skin Lesions                                                   │ True   │ False  │ True   │ 10.34970/559884 │
│ 251 │ Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022) │ True   │ False  │ True   │ 10.34970/587329 │
│ 176 │ Hospital Italiano de Buenos Aires Skin Lesions                      │ True   │ False  │ True   │ 10.34970/432362 │
│ 217 │ Longitudinal overview images of posterior trunks                    │ True   │ False  │ True   │ 10.34970/630662 │
│ 289 │ MSK-1                                                               │ True   │ False  │ True   │ None            │
│ 290 │ MSK-2                                                               │ True   │ False  │ True   │ None            │
│ 288 │ MSK-3                                                               │ True   │ False  │ True   │ None            │
│ 287 │ MSK-4                                                               │ True   │ False  │ True   │ None            │
│ 286 │ MSK-5                                                               │ True   │ False  │ True   │ None            │
│ 163 │ MSKCC Consecutive biopsies across year 2020_cohort                  │ True   │ False  │ True   │ None            │
│ 77  │ Melanocytic lesions used for dermoscopic feature annotations        │ True   │ False  │ True   │ 10.34970/108631 │
│ 215 │ Newly-acquired and longer-existing acquired melanoma and nevi       │ True   │ False  │ True   │ 10.34970/408649 │
│ 218 │ PROVe-AI                                                            │ True   │ True   │ True   │ 10.34970/576276 │
│ 293 │ SONIC                                                               │ True   │ False  │ True   │ None            │
│ 292 │ UDA-1                                                               │ True   │ False  │ True   │ None            │
│ 291 │ UDA-2                                                               │ True   │ False  │ True   │ None            │
│ 285 │ lesions                                                             │ True   │ False  │ False  │ None            │
│ 172 │ screenshot_public_230207                                            │ True   │ False  │ False  │ None            │
└─────┴─────────────────────────────────────────────────────────────────────┴────────┴────────┴────────┴─────────────────┘

isic metadata download --collections 218 -o isic_metadata_218.csv

isic image download -c 218 images


import pandas as pd
metadata_all=pd.read_csv("images/metadata.csv")
training_overlap=pd.read_csv("/sdata/chanwkim/dermatology_datasets/isic/final_training_overlap.csv")
dupcheck=pd.read_csv("/sdata/chanwkim/dermatology_datasets/isic/final_dupcheck.csv")

metadata_all["isic_id"].isin(training_overlap["target_idx"].tolist()).sum()
all_dup=pd.concat([dupcheck["0"], dupcheck["1"], dupcheck["2"], dupcheck["3"], dupcheck["4"]])
all_dup=all_dup[~all_dup.isnull()]
metadata_all["isic_id"].isin(all_dup.tolist()).sum()


python src/MONET/preprocess/glob_files.py \
--input data/proveai/images \
--output data/proveai/images.compact.pkl \
--field images \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input data/proveai/images.compact.pkl \
--output data/proveai/images.compact.uncorrupted.pkl \
--field images

cp data/proveai/images/metadata.csv /sdata/chanwkim/dermatology_datasets/proveai/final_metadata_all.csv

python src/MONET/preprocess/save_as_path.py \
--input data/proveai/images.compact.uncorrupted.pkl \
--field images \
--output /sdata/chanwkim/dermatology_datasets/proveai/final_image