# human-pose-synthesis

This repo implements Unsupervised Person Image Generation with Semantic Parsing Transformation and uses and modifies codes from this [repo](<https://github.com/SijieSong/person_generation_spt>).

### Requrements
Install all dependencies in `requrements.txt`.

### Content
* `data`: define the data leader for the model.
* `model`: define the main model.
* `options`: command line arguments.
* `preprocess`: preprocess the data, obtains the keypoints information.
* `results`: the demo output images.
* `scripts`: bash script to execute the test demo.
* `util`: utilities to get the poses and pose masks.
* `test_demo.py`: for demo.

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
