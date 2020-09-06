## Code Structure


1. Please download and extract selfanime data under folder dataset

2. Assume you extract the self2anime dataset under path "dataset/bundle"


## Run a full model with gpu
python main.py --device=cuda --dataset=bundle


## Run a light model with gpu
python main.py --device=cuda --light True --dataset=bundle


## Testing
- 下载测试脚本　https://github.com/kafaichan/GAN_Metrics-Tensorflow
- 下载权重文件 https://drive.google.com/drive/folders/13_Ec4qsIw6oqv_MRKGrUKrS2qcdYZNU5?usp=sharing, 並把文件放入results/anime/model下 (註: 每次运行測試或训练时，程序只会加载结尾编号最大的模型,建议使用结尾编号为400000的模型，A2B和B2A的生成效果较佳，而结尾编号为688000的模型在A2B取得最佳效果)
- 切換到work目录下, 运行 python main.py --phase=test
- 根据GAN metric repo指示创建文件夹
- 将测试集testA, testB图像分别放入real_source, real_target. 　将results/anime/test/下前缀为A2B_的100幅图像放入fake文件夹
- 运行脚本得出A2B的mean_KID_mean值
- 将测试集testB, testA图像分别放入real_source, real_target. 　将results/anime/test/下前缀为B2A_的100幅图像放入fake文件夹
- 运行脚本得出B2A的mean_KID_mean值




