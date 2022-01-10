# SimpleDepthEstimation

## Introduction

This is an unified codebase for NN-based monocular depth estimation, the framework is based on [detectron2](https://github.com/facebookresearch/detectron2) (with a lot of modifications) and supports both supervised and self-supervised monocular depth estimation methods. The main goal for developing this repository is to help understand popular depth estimation papers, I tried my best to keep the code simple.


## Environment:
1. clone this repo
   ```bash
   SDE_ROOT=/path/to/SimpleDepthEstimation
   git clone https://github.com/zzzxxxttt/SimpleDepthEstimation $SDE_ROOT
   cd $SDE_ROOT
   ```
2. create a new conda environment and activate it
   ```bash
   conda create -n sde python=3.6 
   conda activate sde
   ```
3. install torch==1.8.0 and torchvision==0.9.0 follow the [official instructions](https://pytorch.org/). (I haven't tried other pytorch versions)
4. install other requirements
   ```bash
   pip install -r requirements.txt
   ```


## Data preparation
### KITTI:
Download and extract [KITTI raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php), [refined KITTI depth groundtruth](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip), and [eigen split files](https://github.com/cleinc/bts/tree/master/train_test_inputs), then modify the data path in the config file.


## Training 
```bash
python path/to/project/train.py --num-gpus 2 --cfg path/to/config RUN_NAME run_name
```


## Evaluation
```bash
python path/to/project/train.py --num-gpus 2 --cfg path/to/config --eval MODEL.WEIGHTS /path/to/checkpoint_file
```


## Results:
### KITTI:
|         model          |      type       |                     config                      | abs rel err | sq rel err |  rms  | log rms |  d1   |  d2   |  d3   |
| :--------------------: | :-------------: | :---------------------------------------------: | :---------: | :--------: | :---: | :-----: | :---: | :---: | :---: |
|       ResNet-18        |   supervised    | [link](projects/Supervised/configs/resnet.yaml) |    0.076    |   0.306    | 3.066 |  0.116  | 0.936 | 0.990 | 0.998 |
|   [BTSNet](https://arxiv.org/abs/1907.10326) (ResNet-50)   |   supervised    |  [link](projects/Supervised/configs/bts.yaml)   |    0.062    |   0.259    | 2.859 |  0.100  | 0.950 | 0.992 | 0.998 |
| [MonoDepth2](https://arxiv.org/abs/1806.01260) (ResNet-18) | self-supervised | [link](projects/MonoDepth2/configs/resnet.yaml) |    0.118    |   0.735    | 4.517 |  0.163  | 0.860 | 0.974 | 0.994 |
| [PackNet](https://arxiv.org/abs/1905.02693) | self-supervised | [link](projects/MonoDepth2/configs/packnet.yaml) |    0.107    |   0.762    | 4.577 |  0.159  | 0.884 | 0.972 | 0.992 |


## Demo:
```bash
python tools/demo.py --cfg path/to/config --input path/to/image --output path/to/output_dir MODEL.WEIGHTS /path/to/checkpoint_file
```

**Demo results:**

![](imgs/vis.gif)

## Todo
- [ ] add [Dynamic Motion Learning](https://arxiv.org/abs/2010.16404) (I have implemented it but still buggy, help welcome!)
- [ ] add [Depth From Videos in the Wild](https://openaccess.thecvf.com/content_ICCV_2019/html/Gordon_Depth_From_Videos_in_the_Wild_Unsupervised_Monocular_Depth_Learning_ICCV_2019_paper.html)
- [ ] support more datasets
- [ ] support multi camera

## Reference
- [detectron2](https://github.com/facebookresearch/detectron2)
- [monodepth2](https://github.com/nianticlabs/monodepth2)
- [bts](https://github.com/cleinc/bts)
- [packnet-sfm](https://github.com/TRI-ML/packnet-sfm)
- [motion-learning](https://github.com/google-research/google-research/tree/master/depth_and_motion_learning)