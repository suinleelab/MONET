# MONET (Medical cONcept rETriever)

## Usage

```bash
# clone project
git clone https://github.com/chanwkimlab/MONET
cd MONET

# [OPTIONAL] create conda environment
conda create -n MONET python=3.9
conda activate MONET

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

```bash
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```
