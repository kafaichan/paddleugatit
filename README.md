

1. Please download and extract selfanime data under folder dataset

2. Assume you extract the self2anime dataset under path "dataset/bundle"


## Run a full model with gpu
python main.py --device=cuda --dataset=bundle


## Run a light model with gpu
python main.py --device=cuda --light True --dataset=bundle


3. Testing
'''
  
    a.下载测试脚本　https://github.com/kafaichan/GAN_Metrics-Tensorflow
    b.根据GAN metric repo指示创建文件夹
    c.将测试集testA, testB图像分别放入real_source, real_target. 　将results/anime/test/下前缀为A2B_的100幅图像放入fake文件夹
    d.运行脚本得出A2B的mean_KID_mean值
    e.将测试集testB, testA图像分别放入real_source, real_target. 　将results/anime/test/下前缀为B2A_的100幅图像放入fake文件夹
    f.运行脚本得出B2A的mean_KID_mean值
'''


