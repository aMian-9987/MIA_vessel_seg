# Official Implement of Paper: A novel hepatic vein/portal vein segmentation model for diagnosis and preoperative liver surgical planning
This repo is the official implementation for our paper submitted to Medical Image Analysis.
### Requirements
python 3.6
pytorch 1.8.0
torchvision 0.9.0
Monai Library and some other computational packages.\

## Datasets

* 3D-ircadb dataset can be found at: https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/
* Our dataset will be release after accept.

## Training

Commands for training on the dataset
```
python train_ircadb.py 
```
Commands for pretrain on the dataset
```
python pretrain.py
```
## Testing

Commands for training on the Synapse dataset
``` 
python test.py
