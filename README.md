# Detection of diaebetic retinopathy from fundus images

This project was a part of course project for the CS682 - Introduction to neural networks, fall 2017, course at UMASS. 

This project is to detect diabetic retinopathy from fundus images. Convolutional neural network have been used to detect the presence of diabetic retinopathy from a dataset of Fluorescein Angiography photographs. The dataset was obtained from kaggle where it was provided by EyePacs. A VGG19 network trained on this image achieved an accuracy of 74\% and sensitivity of 77 %

We have experimented with fine-tuning VGG19 and Inception v3 on the retina image dataset and used a cross entropy loss function. The classification task is to detect the presence of diabetic retinopathy. Data pre-processing and augmentation was also done before training the neural network model. 

Used pytorch
