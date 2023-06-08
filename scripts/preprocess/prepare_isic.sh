# isic metadata download --outfile isic_metadata.csv

isic image download images/

isic metadata download --collections 61 -o isic_metadata_61.csv
isic metadata download --collections 74 -o isic_metadata_74.csv
isic metadata download --collections 69 -o isic_metadata_69.csv
isic metadata download --collections 60 -o isic_metadata_60.csv
isic metadata download --collections 71 -o isic_metadata_71.csv
isic metadata download --collections 63 -o isic_metadata_63.csv
isic metadata download --collections 66 -o isic_metadata_66.csv
isic metadata download --collections 65 -o isic_metadata_65.csv
isic metadata download --collections 70 -o isic_metadata_70.csv
isic metadata download --collections 97 -o isic_metadata_97.csv
isic metadata download --collections 162 -o isic_metadata_162.csv
isic metadata download --collections 75 -o isic_metadata_75.csv
isic metadata download --collections 166 -o isic_metadata_166.csv
isic metadata download --collections 168 -o isic_metadata_168.csv
isic metadata download --collections 163 -o isic_metadata_163.csv
isic metadata download --collections 77 -o isic_metadata_77.csv
isic metadata download --collections 170 -o isic_metadata_170.csv
isic metadata download --collections 172 -o isic_metadata_172.csv

# import glob
# import pandas as pd

# metadata_all=pd.read_csv("data/isic/images/metadata.csv")
# collection_file_list=glob.glob("data/isic/isic_metadata_*.csv")
# for collection_file in collection_file_list:
#     collection=pd.read_csv(collection_file)
#     collection_id=collection_file.split('/')[-1].split('.')[0].split('_')[-1]
#     metadata_all["collection_"+collection_id]=0
#     metadata_all["collection_"+collection_id][metadata_all["isic_id"].isin(collection["isic_id"])]=1
# metadata_all["attribution"]=metadata_all["attribution"].str.replace("ViDIR group", "ViDIR Group")
# metadata_all.to_csv("data/isic/images/metadata_collection.csv", index=False)


python src/MONET/preprocess/glob_files.py \
--input data/isic/images \
--output data/isic/images.compact.pkl \
--field images \
--style slash_to_underscore \
--extension .jpg,.jpeg,.png,.tif,.tiff,.gif,.bmp;

taskset -c 0,1,2,3,4,5,6,7 python src/MONET/preprocess/image_sanity_check.py \
--input data/isic/images.compact.pkl \
--output data/isic/images.compact.uncorrupted.pkl \
--field images

cp data/isic/images/metadata_collection.csv /sdata/chanwkim/dermatology_datasets/isic/final_metadata_all.csv
python src/MONET/preprocess/save_as_path.py \
--input data/isic/images.compact.uncorrupted.pkl \
--field images \
--output /sdata/chanwkim/dermatology_datasets/isic/final_image


│ 61  │ Challenge 2016: Test                                         │ True   │ True   │ True   │ None                            │
│ 74  │ Challenge 2016: Training                                     │ True   │ True   │ True   │ None                            │
│ 69  │ Challenge 2017: Test                                         │ True   │ True   │ True   │ None                            │
│ 60  │ Challenge 2017: Training                                     │ True   │ True   │ True   │ None                            │
│ 71  │ Challenge 2017: Validation                                   │ True   │ True   │ True   │ None                            │
│ 63  │ Challenge 2018: Task 1-2: Training                           │ True   │ True   │ True   │ None                            │
│ 66  │ Challenge 2018: Task 3: Training                             │ True   │ True   │ True   │ None                            │
│ 65  │ Challenge 2019: Training                                     │ True   │ True   │ True   │ None                            │
│ 70  │ Challenge 2020: Training                                     │ True   │ True   │ True   │ None                            │
│ 97  │ Collection for ISBI 2016: 100 Lesion Classification          │ True   │ False  │ True   │ None                            │
│ 162 │ Consecutive biopsies for melanoma across year 2020           │ True   │ False  │ True   │ https://doi.org/10.34970/151324 │
│ 75  │ Consumer AI apps                                             │ True   │ False  │ True   │ https://doi.org/10.34970/401946 │
│ 166 │ EASY Dermoscopy Expert Agreement Study                       │ True   │ False  │ False  │ None                            │
│ 168 │ Longitudinal overview images of posterior trunks             │ True   │ False  │ True   │ https://doi.org/10.34970/630662 │
│ 163 │ MSKCC Consecutive biopsies across year 2020_cohort           │ True   │ False  │ True   │ None                            │
│ 77  │ Melanocytic lesions used for dermoscopic feature annotations │ True   │ False  │ True   │ https://doi.org/10.34970/108631 │
│ 170 │ PROVe-AI                                                     │ True   │ True   │ True   │ https://doi.org/10.34970/576276 │
│ 172 │ screenshot_public_230207                                     │ True   │ False  │ False  │ None                            │
└─────┴──────────────────────────────────────────────────────────────┴────────┴────────┴────────┴─────────────────────────────────┘

{61: "Challenge 2016: Test",
74: "Challenge 2016: Training",
69: "Challenge 2017: Test",
60: "Challenge 2017: Training",
71: "Challenge 2017: Validation",
63: "Challenge 2018: Task 1-2: Training",
66: "Challenge 2018: Task 3: Training",
65: "Challenge 2019: Training",
70: "Challenge 2020: Training",
97: "Collection for ISBI 2016: 100 Lesion Classification",
162: "Consecutive biopsies for melanoma across year 2020",
75: "Consumer AI apps",
166: "EASY Dermoscopy Expert Agreement Study",
168: "Longitudinal overview images of posterior trunks",
163: "MSKCC Consecutive biopsies across year 2020_cohort",
77: "Melanocytic lesions used for dermoscopic feature annotations",
170: "PROVe-AI",
172: "screenshot_public_230207"}

