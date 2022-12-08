# CS415 (Computer Vision I) Final Project: From K-means to Deep Learning
## Authors: Giuseppe Concialdi, Giuseppe Stracquadanio and Gaetano Coppoletta - University of Illinois Chicago 

In this work we explored the task of Image Segmentation, conducting different
experiments with several approaches. We first adopted simple clustering methods
to solve the task, such as K-means and Mean-Shift. While these algorithms can
work quite well for a given image, they don’t have generalization capability and
they don’t provide a learning framework. Moreover, they are not able to capture
any semantic about images, so they can’t be used for Semantic Segmentation,
which extends the task of Image Segmentation by associating a semantic label to
each pixel. Thus, we employed Deep Learning models to solve the semantic task,
showing a comparison between different architectures

The provided notebook *clustering.ipynb* contains the code for reproducing our results for *K-means* and *Mean-Shift*.

*train.py* script containts all the code for training and evaluating the provided  Deep Learning models (U-Net, Dilated U-Net and BiSegNet) for Semantic Segmentation.
To see all the available options for the parser, type:
```
python train.py -h
``` 

Example of usage:
```
python train.py --model "unet" --epochs 30 --tensorboard_logdir "output/segmentation/runs/" --save_images_step 1 --validation_step 1
```

*data* folder also contains our smaller version of **Cityscapes** dataset. This dataset only contains 500 images for training and 250 for evaluation, allowing to train the models on hardware with limited capabilities. The number of labels was also reduced from 30 (as in the original **Cityscapes** dataset) to 19. 

*cityscapes.py* provides a Custom PyTorch DataLoader for the used dataset.