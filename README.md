# EGFR_DGMG
 A pretrained model for molecules generation
 
 In this Repo, we show how to fine-tune a pre-trained model and build your model for molecules generation.
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

##Verifying successful installation

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



