# PA-DAN
# XXXXXXXXXXXX


[xxx](https://arxiv.org/abs/xxx)  

If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/1811.12833):

```
@inproceedings{xx,
  title={xxx},
  author={xxx},
  booktitle={xxx},
  year={xxx}
}
```

## Abstract
xxx.


## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 1.1.0
* CUDA 9.0 or higher

### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/seabearlmx/PA-DAN
$ cd PA-DAN
```

### Datasets
By default, the datasets are put in ```<root_dir>/DADatasets```. 

* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
<root_dir>/DADatasets/GTA5/                               % GTA dataset root
<root_dir>/DADatasets/GTA5/images/                        % GTA images
<root_dir>/DADatasets/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/DADatasets/Cityscapes/                         % Cityscapes dataset root
<root_dir>/DADatasets/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/DADatasets/Cityscapes/leftImg8bit/val
<root_dir>/DADatasets/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/DADatasets/Cityscapes/gtFine/val
...
```

### Pre-trained models
Pre-trained models can be downloaded [here](https://github.com/seabearlmx/PA-DAN/releases) and put in ```<root_dir>/padan/pretrained_models```

## Running the code
Please follow the [here](https://github.com/seabearlmx/PA-DAN/releases) to download model.

For evaluation, execute:
```bash
$ cd <root_dir>/padan
$ python test.py --cfg ./configs/padan.yml
```

### Training
For the experiments done in the paper, we used pytorch 1.1.0 and CUDA 9.0. To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance.

By default, logs and snapshots are stored in ```<root_dir>/experiments``` with this structure:
```bash
<root_dir>/experiments/logs
<root_dir>/experiments/snapshots
```

To train PA-DAN:
```bash
$ cd <root_dir>/padan
$ python train.py --cfg ./configs/padan.yml

```

### Testing
To test PA-DAN:
```bash
$ cd <root_dir>/padan
$ python test.py --cfg ./configs/padan.yml
```

## Acknowledgements
This codebase is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet) and [AdvEnt](https://github.com/valeoai/ADVENT).

## License
PA-DAN is released under the [MIT license](./LICENSE).
