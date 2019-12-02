# human-pose-synthesis

This repo implements Unsupervised Person Image Generation with Semantic Parsing Transformation and uses and modifies codes from this [repo](<https://github.com/SijieSong/person_generation_spt>).

### Requrements
Install all dependencies in `requrements.txt`.

### Content
* `data`: precessed MPII csv data.
* `model`: implementation of various neural networks.
* `person_generation`: use per-trained model to generate pose-guided human synthesis images.
* `preprocess`: preprocess the data, obtains the keypoints information.
* `utils`: some helper methods.

### Reference

Unsupervised Person Image Generation with Semantic Parsing Transformation (CVPR 2019, oral).<br>
[Sijie Song](https://sijiesong.github.io/), [Wei Zhang](https://wzhang34.github.io/), [Jiaying Liu](http://icst.pku.edu.cn/struct/people/liujiaying.html), [Tao Mei](https://taomei.me/)

* [Project page](<http://39.96.165.147/Projects/SijieSong_cvpr19/CVPR19_ssj.html>)
* [Paper](<https://arxiv.org/abs/1904.03379>)
* [Supplementary](<http://39.96.165.147/Projects/SijieSong_cvpr19/files/supp.pdf>)

### Related projects

* [person_generation_spt](<https://github.com/SijieSong/person_generation_spt>)
* [OpenPose](<https://github.com/CMU-Perceptual-Computing-Lab/openpose>)
* [Detectron2](<https://github.com/facebookresearch/detectron2>)
