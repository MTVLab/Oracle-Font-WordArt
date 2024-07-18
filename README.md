## 1.甲骨文创意文字(OBI-Designer)

**摘要:** 甲骨文艺术字合成是指结合古文字的历史美感和人工智能技术，创造出既有文化底蕴又具视觉冲击力的艺术作品。目前，甲骨文艺术字的创作主要依靠专家知识，效率低且耗时长。在本文中，我们提出了一种zero-shot的方法自动地创造甲骨文艺术字。OBI-Designer依赖于近期的大规模预训练的语言-视觉模型的显著能力，通过视觉方式提取文本概念，来进行甲骨文艺术字合成。具体来说，OBI-Designer可以分为两个阶段，分别是字形合成和纹理合成。字形合成介绍了一种基于区间分数匹配的语义驱动图像矢量化方法，通过优化字符轮廓来生动表达文本概念，同时使用额外的损失函数保证字符的可读性和一致性。纹理合成基于ControlNet和Lora模型对字形合成的结果进行风格化，以得到最终的甲骨文艺术字。广泛的实验证明了所提出的方法的有效性。OBI-Designer进一步推动了甲骨文艺术字合成与人工智能的结合。

**Abstract:** Artistic typography synthesis of oracle bone inscriptions refers to combining the historical aesthetic of ancient characters with artificial intelligence technology to create artistic works that are both culturally profound and visually impactful. Currently, the creation of artistic typography of oracle bone inscriptions relies heavily on expert knowledge, resulting in low efficiency and long processing times. In this paper, we propose a zero-shot method for automatically creating oracle bone inscriptions artistic typography. OBI-Designer relies on the significant capability of recent large-scale pre-trained language-vision models to visually extract textual concepts. Specifically, OBI-Designer can be divided into two stages: glyph synthesis and texture synthesis. Glyph synthesis introduces a semantic-driven image vectorization method based on interval score matching, optimizing character outlines to vividly express textual concepts while using additional loss functions to ensure the readability and consistency of characters. Texture synthesis stylizes the results of glyph synthesis using ControlNet and Lora models to achieve the final artistic typography of oracle bone inscriptions. Extensive experiments have demonstrated the effectiveness of the proposed method. OBI-Designer further advances the integration of artistic typography of oracle bone inscriptions with artificial intelligence.

### 1.1 艺术字生成流程图

1. 文字字形设计：甲骨文作为象形文字，结构复杂，笔画繁多且单个字符具备完整的含义，若以整个字符进行形变，会导致变形后的字符不可识别，因此以**字符的局部作为形变的对象**。
1. 利用扩散模型（SD）和DDIM反演进行区间分数匹配，使文字轮廓与prompt提示语义相匹配。

<img src="asset/pipeline.png" alt="pipeline" style="zoom: 50%;" align="left" />

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

<img src="asset/example.png" alt="生肖" style="zoom: 100%;" align="left"/>

### 2.3 消融实验

#### 2.3.1 CLIP、SDS和ISM的对比

对比不同的对大规模预训练视觉-语言大模型进行蒸馏的方法。

<img src=".\asset\loss_fn.png" alt="loss_fn" style="zoom:60%;" align="left"/>

#### 2.3.2 控制点数量选择的对比（默认控制点or预设控制点）



