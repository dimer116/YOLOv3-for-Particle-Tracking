# YOLOv3-for-Particle-Tracking

This repository contains the code for the Bachelor's thesis: 
Karakterisering och spårning av nanopartiklar med djupinlärning
(*eng.* Charaterization and tracking of nanoparticles using deep learning)
by  Arash Darakhsh, Edvin Johansson, Simon Nilsson, Sanna Persson and Rickard Ström at Chalmers University of Technology

The code for simulation will shortly be updated with further clarifications to improve ease of use.

## How to use the code

### Download the repository and weights
To clone the repository run:
```bash
git clone https://github.com/Deep-learning-for-particle-tracking/YOLOv3-for-Particle-Tracking.git
```
The weights are available at this [link](https://www.kaggle.com/sannapersson/weights-particle-tracking-yolov3) at Kaggle. 
The source code and mod can also be downloaded in the release. In the data-folder in the release there are two example experimental images. 

### Install requirements
```bash
pip install requirements.txt
```

### Inference with the model
Place the weights in the model folder and the image for inference in npy-format in the data-folder. Run 
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
to read about the training parameters to input them in the terminal. 
To train the model run 
```python
python train.py 
```
A few examples of how to structure the training data is also found in the training_data folder. 

### Simulate images
The code for simulating the images is found in the simulation folder.

