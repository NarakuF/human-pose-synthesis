# human-pose-synthesis
This is a project for COMS 4995 Deep Learning. We propose a new task, which is text guided pose synthesis on large scale of human activities. We first analyze the state-of-the-art model for pose keypoints estimation, the power of human semantic parsing, and existing text guided image synthesis. Further, we proposed an approach that can solve our target task with 3 stages. We then talked about the potential improvement and drawbacks of our model.

## Contributors
* [Guandong Liu](https://github.com/NarakuF)
* [Yue Wan](https://github.com/yuewan2)

## Requirements
Install all dependencies in `requrements.txt`.

## Contents
* `data`: processed MPII csv data.
* `model`: implementation of various neural networks, including annotation classifier, and conditional GAN.
* `utils`: some helper methods to process images and texts.
* `main.py`: main code to run the whole pipeline.
* `pose_dataset.py`: customized dataset and dataloader using PyTorch.
* `train*`: training code for different neural networks.

## References and Related Projects
* [Unsupervised Person Image Generation with Semantic Parsing Transformation](<https://github.com/SijieSong/person_generation_spt>)
* [OpenPose](<https://github.com/CMU-Perceptual-Computing-Lab/openpose>)
* [Detectron2](<https://github.com/facebookresearch/detectron2>)
* More references are cited in the project report.