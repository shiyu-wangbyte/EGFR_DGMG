# EGFR_DGMG
 A pretrained model for molecules generation
 
 In this repo, we show how to fine-tune a pre-trained model and build your own model for molecules generation.

 For more information, click [Here](https://docs.dgl.ai/tutorials/models/3_generative_model/5_dgmg.html)

## Installation
### Requirements

we test all script on opensuse 15.1.

```
conda create my_dgl python=3.7
conda activate my_dgl
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c rdkit rdkit==2018.09.3
conda install -c dglteam dgl
pip install dgllife
```

## Verifying successful installation

```python
import dgllife
import dgl
import torch
import rdkit

print(torch.__version__)
# 1.7.0
print(dgl.__version__)
# 0.6.1
print(rdkit.__version__)
# 2020.09.1
print(dgllife.__version__)
# 0.2.8
```

More information about installation, please check:

[Install pytorch](https://pytorch.org/get-started/locally/)

[Install dgl](https://www.dgl.ai/pages/start.html)

[Install dgllife](https://lifesci.dgl.ai/index.html)

[Install rdkit](https://www.rdkit.org/docs/Install.html)

# Usage

## step 1: Preprocessing your own data

Preprocessing additional data for DGMG model.

'''
python preprocess.py -d EGFR -m ZINC -tf ./EGFR_data/EGFR_train.txt -vf ./EGFR_data/EGFR_val.txt
'''

## Step 2: Training or fine-tuning

Training or fine-tuning DGMG model for molecule generation.

The script will save model each 50 epochs!

'''
python fine_tune.py -d EGFR -m ZINC -o canonical -tf ./EGFR_data/EGFR_DGMG_train.txt -vf ./EGFR_data/EGFR_DGMG_val.txt
'''

## Step 3: Generating molecules

Generate molecules with pretrained model or fine-tuned model.

Just use a pre-trained model:

'''
python generate_mols.py

python generate_mols.py -m ZINC
'''

Use a fine-tuning model

'''
python generate_mols.py -d EGFR -p ./saved_model/EGFR/50_checkpoint.pth -s ./saved_model/EGFR/settings.txt
'''

### Cite
```
@article{,
    title={},
    author={},
    year={2021},
    journal={}
}
```

### Acknowledge

[Dgllife](https://github.com/awslabs/dgl-lifesci)

[Dgl](https://github.com/dmlc/dgl)
