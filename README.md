python projects/Supervised/train.py --num-gpus 2 --cfg configs/Supervised/res.yaml RUN_NAME baseline
python projects/Supervised/train.py --num-gpus 1 --cfg configs/Supervised/res.yaml --eval MODEL.WEIGHTS /path/to/checkpoint_file

python projects/MonoDepth2/train.py --num-gpus 2 --cfg configs/MonoDepth2/resnet.yaml RUN_NAME baseline
python projects/MonoDepth2/train.py --num-gpus 1 --cfg configs/Supervised/debug2.yaml --eval MODEL.WEIGHTS /path/to/checkpoint_file

python projects/MotionLearning/train.py --num-gpus 2 --cfg configs/MotionLearning/resnet.yaml RUN_NAME baseline
python projects/MotionLearning/train.py --num-gpus 1 --cfg configs/MotionLearning/resnet.yaml --eval MODEL.WEIGHTS /path/to/checkpoint_file