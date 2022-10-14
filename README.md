# Neural Space-filling Curves
Official Pytorch implementation of the ECCV 2022 paper.

[[Paper](https://arxiv.org/abs/2204.08453)] [[Project Page](https://hywang66.github.io/publication/neuralsfc)] 

Top: Neural SFC; Bottom: Hilbert curve.

![Comparision](https://hywang66.github.io/publication/neuralsfc/img_00630_neural_hilbert_vis.gif)



## Requirements

First, create a new virtual environment (conda or virtualenv).

Then

```bash
pip install -r requirements.txt
```
Please follow https://pytorch.org/get-started/locally/ to install **pytorch** and https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to install **torch_geometric** properly.

Our code is tested under **Python 3.9.13**, **pytorch 1.11.0**, **cuda 11.3**, and **torch_geometric(pyg) 2.0.4**.

Our specific installation follows:

```bash
cd path/to/project
conda create -n neuralsfc python=3.9.13
conda activate neuralsfc
pip install -r requirements.txt
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
```
but it should also work in other environments.

## Training
In the folder `config`, we provide configurations for training NeuralSFCs on MNIST, Fashion-MNIST, and FFHQ 32x32 datasets. 
For each dataset, there are two configurations, `*_ac.yaml` for the autocorrelation objective and `*_lzwl.yaml` for the LZW encoding length objective.

All datasets are downloaded automatically when running the training script.

For example, to train a NeuralSFC on MNIST with the autocorrelation objective, run


```bash
cd path/to/project
export PYTHONPATH=".:$PYTHONPATH"
python neuralsfc/train.py --cfg configs/mnist_ac.yml
```


## Evalution

In this section, we provide commands to evaluate the **LZW encoding length** of the trained NeuralSFCs models. 

Our pretrained models for MNIST, Fashion-MNIST, and FFHQ 32x32 datasets are available [here](https://drive.google.com/drive/folders/1BNGSCOBw3Xe0qzCPcM8mWEk8vp-QGUV4?usp=sharing). You can also train your own models by following the instructions in the previous section.

To evaluate the LZW encoding length of a saved model, you will need both the model checkpoint (\*.pt) and configuration file (\*.yml). For example, to evaluate the LZW encoding length of the pretrained model on MNIST, run

```bash
cd path/to/project
export PYTHONPATH=".:$PYTHONPATH"
python neuralsfc/train.py --eval --cfg configs/mnist_lzwl.yml --load_path path/to/mnist_lzwl.pt
```

You will see some outputs like

```
avg length: 171.1736, std: 33.3496786047482
```

A folder `eval` will be also created in the same directory as the model checkpoint. The LZW encoding length will be saved in `eval/{checkpoint_name}_length.txt`. The generated SFC will be saved in `eval/{checkpoint_name}_neuralsfc.npy` or `eval/{checkpoint_name}_neuralsfc_{label}.npy`, depending on whether the dataset is conditional or not.


## BibTeX

```
@article{wang2022neural,
  title={Neural Space-filling Curves},
  author={Wang, Hanyu and Gupta, Kamal and Davis, Larry and Shrivastava, Abhinav},
  journal={arXiv preprint arXiv:2204.08453},
  year={2022}
}
```