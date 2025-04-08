# MONET (Medical cONcept rETriever)

[MONET](https://doi.org/10.1038/s41591-024-02887-x) is an image-text foundation model trained on 105,550 dermatological images paired with natural language descriptions from a large collection of medical literature. MONET can accurately annotate concepts across dermatology images as verified by board-certified dermatologists, competitively with
supervised models built on previously concept-annotated dermatology datasets of clinical images. MONET enables AI transparency across the entire AI system development pipeline from building inherently interpretable models to dataset and model auditing.

- [Paper](https://doi.org/10.1038/s41591-024-02887-x)
- [GitHub](https://github.com/suinleelab/MONET)
- [BibTex](#citation)

## Getting started

### Install

To install the required packages, run the following bash commands:

```bash
# clone project
git clone https://github.com/suinleelab/MONET
cd MONET

# [OPTIONAL] create conda environment
conda create -n MONET python=3.9.15
conda activate MONET

# install PyTorch according to instructions at https://pytorch.org/get-started/ v.1.13.0 was used during development.
# example: conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# install other required python packages
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Initialize model

#### Using original openai CLIP implementation

```bash
import clip

def get_transform(n_px):
    def convert_image_to_rgb(image):
        return image.convert("RGB")
    return T.Compose(
        [
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

model, preprocess = clip.load("ViT-L/14", device="cuda:0", jit=False), get_transform(n_px=224)
model.load_state_dict(torch.hub.load_state_dict_from_url("https://aimslab.cs.washington.edu/MONET/weight_clip.pt"))
model.eval()
```

#### Using huggingface CLIP implementation

```bash
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor_hf = AutoProcessor.from_pretrained("chanwkim/monet")
model_hf = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet")
model_hf.to("cuda:0")
model_hf.eval()
```

### Usage

We provide jupyter notebooks to demonstrate how to use MONET for automatic concept annotation and various transparency tasks such as data auditing, model auditing, and inherently interpretable model building.

- Automatic concept annotation: `tutorial/automatic_concept_annotation.ipynb`
- Data auditing: `tutorial/data_auditing.ipynb`
- Model auditing: `tutorial/model_auditing.ipynb`
- Inherently interpretable model building: `tutorial/inherently_interpretable_model_building.ipynb`

## MONET Training data

For code to download and preprocess the training data, please refer to the following scripts:

```bash
scripts/preprocess/preprocess_pubmed.sh
scripts/preprocess/preprocess_pdf.sh
```

The filtered PubMed subset of the training data is available [here](https://aimslab.cs.washington.edu/MONET/pubmed_data.pkl). You can load the data using the following code:

```python
import pandas as pd
pd.read_pickle("https://aimslab.cs.washington.edu/MONET/pubmed_data.pkl")
```

## Training / Evaluation

Code for preprocessing data and training MONET is available in `src` folder. Code used for evaluation in our paper is available in `experiments` folder.

## Citation

```bibtex
@article{kim2024transparent,
    title = {Transparent medical image AI via an image–text foundation model grounded in medical literature},
    author = {Kim, Chanwoo and Gadgil, Soham U. and {DeGrave}, Alex J. and Omiye, Jesutofunmi A. and Cai, Zhuo Ran and Daneshjou, Roxana and Lee, Su-In},
    journal={Nature Medicine},
    volume = {30},
    number = {4},
    year={2024},
    pages = {1154--1165},
    doi={10.1038/s41591-024-02887-x},
    issn = {1546-170X},
    url={https://doi.org/10.1038/s41591-024-02887-x}
}
```
