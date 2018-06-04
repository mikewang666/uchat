作者：拾荒者000
链接：https://www.nowcoder.com/discuss/66114?type=0&order=0&pos=103&page=1
来源：牛客网

楼主秋招过程，也没面几家（8月低到9月中旬面试安排比较多），9月15号之前把公司定了。月底就把三分寄出去了，找的比较随意。国庆节之后，一家没面，把所有的面试都推了，进入二面的也推了。
现在把我面试的公司的面试情况总结一下（以下仅代表个人经验和观点）：
1、简历上的项目非常重要，对一些细节一定要熟悉，尤其是项目中用到的算法。
2、基础很重要！！！
图谱科技面试

1、  梯度下降：为什么多元函数在负梯度方向下降最快？

数学证明题，和专业无关-->设多元函数及多元函数的单位向量，再求多元函数在单位向量上的方向导数



2、  Sigmoid激活函数为什么会出现梯度消失？Sigmoid函数导数的最大值出现在哪个值？-->（x=0处）                       
        ReLU激活函数为什么能解决梯度消失问题？
3、  Softmax是和什么loss function配合使用？-->多项式回归loss 

该loss function的公式？

4、  （以xx loss function）推导BP算法？

5、  CNN中，卷积层的输入为df*df*M（weight,height,channel），输出为df*df*N（或输出为df*df*M），卷积核大小为dk*dk时，请问由输入得到输出的计算量为多少？题中默认stride=1    

    计算量-->浮点数计算量：49HWC^2，27HWC^2-->会把滤波器用到每个像素上，即为长x宽x可学习的参数个数

6、  说一下dropout的原理？若在训练时，以p的概率关闭神经元，则在预测（测试）的时候概率1-p怎么使用？https://yq.aliyun.com/articles/68901

测试时，被dropout作用的层，每个神经元的输出都要乘以（1-p）à使训练和测试时的输出匹配

7、  传统机器学习是否了解？

8、  说一下作项目时遇到的困难？

9、  表达式为max(x,y)的激活函数，反向传播时，x、y上的梯度如何计算à

答：较大的输入的梯度为1，较小输入的梯度为0；即较小的输入对输出没有影响；另一个值较大，它通过最大值运算门输出，所以最后只会得到较大输入值的梯度。à这也是最大值门是梯度路由的原因。

前向传播时，最大值往前传播；反向传播时，会把梯度分配给输入值最大的线路，这就是一个梯度路由。

地平线机器人面试

1、  检测框架faster rcnn是怎样的一个框架？à这里回答了faster rcnn的过程

2、  Faster rcnn中，ROI pooling具体如何工作（怎么把不同大小的框，pooling到同样的大小）？

RoIPool首先将浮点数值的RoI量化成离散颗粒的特征图，然后将量化的RoI分成几个空间的小块（spatial bins），最后对每个小块进行max pooling操作生成最后的结果。

3、优化代码的方法：多线程等à多线程比单线程快

3、  深度学习那个项目做的方法没有创新点；深度学习项目，数据集要自己做，检测方法要创新à自己制作数据集并添加新层（新的激活函数maxout）

4、  每个项目的衡量指标；如：（1）双目追踪能检测的目标最小是多大à能检测的最小目标是根据实时图像中最大的目标而定的，设定目标面积小于最大的目标的面积的1/5是不能检测的。

（2）深度学习中的指标mAP等（衡量模型好坏的指标？）平均精度（mAP）如何计算的？http://blog.csdn.net/timeflyhigh/article/details/52015163

http://blog.csdn.net/Relocy/article/details/51453950

目标检测的指标：识别精度，识别速度，定位精度

A、目标检测中衡量识别精度的指标是mAP（mean average precision）。多个类别物体检测中，每一个类别都可以根据recall和precision绘制一条曲线，AP就是该曲线下的面积，mAP是多个类别AP的平均值。

B、  目标检测评价体系中衡量定位精度的指标是IoU,IoU就是算法预测的目标窗口和真实的目标窗口的交叠（两个窗口面积上的交集和并集比值），Pascal VOC中，这个值是0.5（已被证明相对宽松）。

机器学习中评价指标： Accuracy、 Precision、Recall

6、 熟悉基本的图像处理算法和图像处理方法（如图像矫正）

5、  Caffe中具有哪些层，如data layer、image data layer、softmaxwithloss（还有其它loss）

6、  训练网络时，如果要每个batch中每种类别的图象数固定（按自己定的取），则该怎么做？（训练时，每个batch都是随机从数据集中抽取一定数量的样本）。

7、  立体匹配有哪些方法？收藏的链接

8、  混合高斯模型（GMM）是怎么样的？à原理和公式

混合高斯模型是无监督学习à可用于聚类

http://www.cnblogs.com/mindpuzzle/archive/2013/04/24/3036447.html

http://blog.csdn.net/wqvbjhc/article/details/5485242

混合高斯模型使用K（基本为3到5个）个高斯模型来表征图像中各个像素点的特征,在新一帧图像获得后更新混合高斯模型, 用当前图像中的每个像素点与混合高斯模型匹配,如果成功则判定该点为背景点,将其归入该模型中，并对该模型根据新的像素值进行更新，若不匹配，则以该像素建立一个高斯模型，初始化参数，代理原有模型中最不可能的模型。最后选择前面几个最有可能的模型作为背景模型，为背景目标提取做铺垫。

9、  光流法？

http://blog.csdn.net/carson2005/article/details/7581642

http://www.cnblogs.com/xingma0910/archive/2013/03/29/2989209.html

光流法，通过求解偏微分方程求的图像序列的光流场，从而预测摄像机的运动状态

10、              Kalman滤波器的原理？

Kalman滤波器最重要的两个过程：预测和校正

 http://blog.csdn.net/carson2005/article/details/7367135

11、              需要熟悉简历上项目写的每个算法的具体过程甚至公式；；；以及是否对算法进行改进，即修改OpenCV的源码

12、              编程：合并两个单调不减的链表，并保证合并后的链表也是单调不减的？

好未来

1、  LeetCode第一题“TwoSum”

2、  通过简单示例，详细解释ROC曲线，要求给出必要的公式推导。

3、  给出LR（逻辑回归）算法的cost function公式的推导过程。

4、  目标检测时，输入的是视频时，如何进行检测？视频中有很多无用的帧（不包含要检测的目标等）-->人工分割视频、每隔一定数量的帧进行检测

5、  项目。

美团

一面：

1、  faster rcnn中ROI pooling 是不是第一次用？-->第一次用是在fast rcnn中

2、  在检测物体时，物体只有少部分在图像中时，是否检测？

系统检测的最小目标为16*16；当部分在图像中时，也对其进行检测（这里有一个阈值，当目标的面积占图像的面积比小于1/5时，不检测；否则就检测（这个思路是从单路分类那来的））

3、  双目视觉中，立体校正如何进行？-->立体标定得出俩摄像机的旋转和平移矩阵，然后在对左右图像进行校正，使其行对齐。

4、  Kalman滤波器是否有运动方程？没建立运动方程，直接将物体轮廓的外接矩形的中心点作为初始化追踪点进行后续追踪。

5、  双目视觉中，光流法用的哪一种？L-K光流是稠密的还是稀疏的？

金字塔l-k光流，其计算的是稀疏特征集的光流。

非金字塔l-k光流(原始的l-k光流)计算的是稠密光流。

http://blog.sina.com.cn/s/blog_15f0112800102wjai.html

二面：

1、  Fatser rcnn与rcnn的不同？-->fatser rcnn是端到端；rcnn不是端到端

2、  Rcnn、fatse rcnn、fatser rcnn、mask rcnn的原理？

mask rcnn-->在fatser rcnn的基础上对ROI添加一个分割的分支，预测ROI当中元素所属分类，使用FCN进行预测；

具体步骤：使用fatser rcnn中的rpn网络产生region proposal（ROI），将ROI分两个分支：（1）fatser rcnn操作，即经过ROI pooling 输入fc进行分类和回归；（2）mask操作，即通过ROIAlign校正经过ROI Pooling之后的相同大小的ROI，然后在用fcn进行预测（分割）。

ROIAlign产生的原因：RoI Pooling就是实现从原图区域映射到卷积区域最后pooling到固定大小的功能，把该区域的尺寸归一化成卷积网络输入的尺寸。在归一化的过程当中，会存在ROI与提取到的特征不对准的现象出现，由于分类问题对平移问题比较鲁棒，所以影响比较小。但是这在预测像素级精度的掩模时会产生一个非常的大的负面影响。作者就提出了这个概念ROIAlign，使用ROIAlign层对提取的特征和输入之间进行校准。

ROIAlign方法：作者用用双线性插值（bilinear interpolation）在每个RoI块中4个采样位置上计算输入特征的精确值，并将结果聚合（使用max或者average）。

Lmask为平均二值交叉熵损失。

实例分割的目的是区分每一个像素为不同的分类而不用区别不同的目标。实例分割就是要在每一个像素上都表示出来目标所属的具体类别。

3、  介绍resnet和GoogLeNet中的inception module的结构？

ResNet 主要的创新在残差网络，其实这个网络的提出本质上还是要解决层次比较深的时候无法训练的问题。这种借鉴了Highway Network思想的网络相当于旁边专门开个通道使得输入可以直达输出，而优化的目标由原来的拟合输出H(x)变成输出和输入的差H(x)-x，其中H(X)是某一层原始的的期望映射输出，x是输入。

à优化后

优化后的结构新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量

Resnet  http://blog.csdn.net/mao_feng/article/details/52734438

Inception module:

共有四个版本。网上搜inception v4就会出现v1-v4

http://www.voidcn.com/article/p-zglerubc-ty.html   （重点）

http://blog.csdn.net/u010025211/article/details/51206237

http://blog.csdn.net/sunbaigui/article/details/50807418

inception v1中用1*1的卷积à降维

Inception v2（BN-inception）在v1的基础上增加BN层，同时将5*5的卷积核替换成两个3*3的卷积核（降低参数数量，加速计算）

Inception v3见博客http://www.voidcn.com/article/p-zglerubc-ty.html

v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这第一个样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性

v3使用的是RMSProp优化方法

inception v4-->16年imagenet分类的第一名

http://blog.csdn.net/lynnandwei/article/details/53736235

由链接中的图可以看出，v4包含v3和v2

4、  yolo和ssd？

5、  Fatser rcnn不能检测小目标的原因？

6、  在训练好的模型数据里， 如何添加识别错误的数据，在进行训练呢？

方法一：直接往lmdb数据里添加，再次重新训练；

方法二：把你的proto里datalayer改成用image data layer 然后把需要添加的图像路径写到list文件里，然后fine tune你的网络

RoKid机器人

1、  Adaboost算法？

2、  逻辑回归实现多分类？

3、  Fatser rcnn中，如何处理背景候选框和含有目标的候选框不平衡问题？

4、  SVM的核函数的作用？

5、其他就是和项目相关的问题？

数码视讯

1、  Canny边缘检测算法的过程？

2、  常用的局部特征和全局特征？

3、LDA原理？
除了上面的公司之外，还有顺丰科技、苏宁、恒润科技，这三个公司问项目相关的比较多，还是要了解自己的项目以及一些相关的基础知识。
面试的水，总结的也水，觉得有用的就看看，不喜勿喷！！！
