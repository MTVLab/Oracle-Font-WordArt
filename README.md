## 1.甲骨文创意文字

### 1.1 艺术字生成流程图

1. 文字字形设计：甲骨文作为象形文字，结构复杂，笔画繁多且单个字符具备完整的含义，若以整个字符进行形变，会导致变形后的字符不可识别，因此以**字符的局部作为形变的对象**。
1. 利用扩散模型（SD）和DDIM反演进行区间分数匹配，使文字轮廓与prompt提示语义相匹配。

![pipeline](asset/pipeline.png)

### 1.2 环境配置（Ubuntu 20.04）

1. 克隆仓库

   ```shell
   # 需要配置ssh
   git clone git@github.com:MTVLab/Oracle-Font-WordArt.git
   cd Oracle-Font-WordArt
   ```

2. 创建conda虚拟环境和安装工具包

   ```sh
   conda create --name word python=3.8.15
   conda activate word
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   conda install -y numpy scikit-image
   conda install -y -c anaconda cmake
   conda install -y -c conda-forge ffmpeg
   pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom freetype-py shapely
   pip install opencv-python==4.5.4.60  
   pip install kornia==0.6.8
   pip install shapely
   ```
   
3. 安装diffusers库（**使用扩散模型必须要了解的库**）

   ```sh
   pip install diffusers
   pip install transformers scipy ftfy accelerate
   ```

   

4. 安装diffvg库（**Linux环境下安装，windows需要配置c++的开发环境，太麻烦**）

   ```sh
   git clone https://github.com/BachiLi/diffvg.git
   cd diffvg
   git submodule update --init --recursive
   python setup.py install
   ```

5. 预训练权重：链接：https://pan.baidu.com/s/1tZVC8xtrSYkiNSJmAFElYw  提取码：8tza 

## 2.初步测试结果

### 2.1 数据集

实验以清华大学陈楠教授和汉仪字库共同开发完成的汉仪陈体甲骨文[^1]为例。

[^1]:[汉仪陈体甲骨文-汉仪字库 (hanyi.com.cn)](https://www.hanyi.com.cn/productdetail.php?id=2638)

### 2.2 生肖甲骨文创意文字

图中第1、2行为甲骨文字，第2、3行为字形变化后的文字，第4、5行为增加纹理后的文字。

<img src="asset/example.png" alt="生肖" style="zoom: 100%;" />

### 2.3 消融实验

#### 2.3.1 不同的图文匹配方法的比较

<img src=".\asset\loss_fn.png" alt="loss_fn" style="zoom:60%;" />



