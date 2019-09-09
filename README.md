# DTA

<p align="center"> 
<img src="./figures/dta_figure.png">
</p>

Official implementation of [Drop to Adapt: Learning Discriminative Features for Unsupervised Domain 
Adaptation](./main.py), to be presented at ICCV 2019.

## Setup

#### Environment
Create a new python virtual environment, and install the packages specified in ```requirements.txt```:

```
pip install -r requirements.txt
```

The code has been tested on Python 3.6.8 with CUDA 9.0 on 4 Titan Xp GPUs.

#### Download Data
Download the [VisDA-2017 Classification](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
data into ```./data```:

```
cd ./data
wget http://csr.bu.edu/ftp/visda17/clf/train.tar
tar xvf train.tar

wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar  
```

## Running the code

We provide two basic configurations under ```./configs/```
for ResNet-101 and ResNet-50 models. 
```
python main.py --config_path ./configs/resnet101_dta_vat.json
python main.py --config_path ./configs/resnet50_dta_vat.json
```

Hyper-parameters can be changed directly from the command line as well. 
For example, to run the ResNet-101 model without VAT:

```
python main.py --config_path ./configs/resnet101_dta_vat.json --use_vat False
```

Tensorboard logs are saved under the path specified in ```./$experiment_dir/$experiment_description```.

## License

## Citation
