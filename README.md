## SAGAN:Self Attention Generative Adversarial Network的TF实现
---

1. [所需环境 Environment](#所需环境)
2. [注意事项 Cautions](#注意事项)
3. [训练步骤 How2train](#训练步骤)

1.所需环境
numpy==1.19.5
scikit-image==0.18.1
opencv-contrib-python==4.5.1.48
tensorflow-gpu==2.5.1  
tensorflow-datasets==4.4.0  

2.注意事项
该手写SAGAN将生成器、对抗器实现为对称结构
自定义实现了各谱归一化层、注意力层

3.运行train.py即可开始训练。
以cars196数据集为生成目标进行训练

