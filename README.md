# YOLOv3-for-Particle-Tracking

This repository contains the code for Bachelor's thesis: Karakterisering och spårning av nanopartiklar med djupinlärning
by  Arash Darakhsh, Edvin Johansson, Simon Nilsson, Sanna Persson and Rickard Ström at Chalmers University of Technology


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
The weights can be downloaded from the following link [link](https://www.kaggle.com/sannapersson/weights-particle-tracking-yolov3)

### Inference with the model
Place your image in npy-format in the data-folder. Run 
```python
python detect_on_patches.py
```
For information on flags and arguments run:
```python
python detect_on_patches.py --help
```

### Train the model
An example dataset can be found on Kaggle of 448x488 images

