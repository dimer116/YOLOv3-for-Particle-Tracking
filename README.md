# YOLOv3-for-Particle-Tracking

This repository contains the code for the Bachelor's thesis: Karakterisering och spårning av nanopartiklar med djupinlärning
by  Arash Darakhsh, Edvin Johansson, Simon Nilsson, Sanna Persson and Rickard Ström at Chalmers University of Technology

The code for simulation will shortly be updated with further clarifications to improve ease of use.

## How to use the code

### Download the repository
```bash
git clone https://github.com/Deep-learning-for-particle-tracking/YOLOv3-for-Particle-Tracking.git
```
### Install requirements
```bash
pip install requirements.txt
```

### Download model weights
The weights can be downloaded from the following [link](https://www.kaggle.com/sannapersson/weights-particle-tracking-yolov3)
Place the weights in the model folder.

### Inference with the model
Place your image in npy-format in the data-folder. Run 
```python
python detect_on_patches.py
```
For information on flags and arguments run:
```python
python detect_on_patches.py --help
```
There are a couple of example experimental images in the folder data which you can test the model on. 

### Train the model
An example dataset can be found on Kaggle of 448x448 images (link: coming soon). Change the configuration for training in the config.py file or
run
```python
python train.py --help
```
to read about the training parameters. 
To train the model run 
```python
python train.py 
```
A few examples of how to structure the training data is also found in the training_data folder. 

### Simulate images
The code for simulating the images is found in the simulation folder.

