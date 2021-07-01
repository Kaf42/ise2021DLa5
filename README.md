# 深度学习A5组作业仓库



## 1.使用vae网络训练数据集：

​	运行vae_net.py
   训练次数Epohs=1000,batchsize=10,训练时间约为2.5h（训练效果差，弃用该网络）

## 2.使用pix2pix网络训练数据集

-   (1)运行pix2pix.py使用pix2pix网络训练数据集
        训练次数Epohs=200,batchsize=10,训练时间约为20h
-  (2)运行test.py使用训练好的模型对测试集进行预测（pix2pix.py训练的模型保存在saved_models/facades/gan_model里）
- （3)使用windows在带的画图软件画出房屋建筑的轮廓图，运行test_selfdraw.py对轮廓图进行预测。

## 3.训练及预测流程

-   (1)运行pix2pix_cutout.py使用优化后的pix2pix网络训练数据集
        训练次数Epohs=50,batchsize=10,训练时间约为7h
- (2)运行test.py使用训练好的模型对测试集进行预测（pix2pix_cutout.py训练的模型保存在saved_models/facades/gancut_model里）
- （3)使用windows在带的画图软件画出房屋建筑的轮廓图，运行test_selfdraw.py对轮廓图进行预测。

## 4.组员及分工

- 作者1：闫嘉豪 效果改进(数据增强尝试), 数据集下载，代码实现，模型保存
- 作者2：肖俊杰 效果改进(数据增强尝试) , 训练过程分析, 数据集分析,海报调整
- 作者3：江潇涛 海报的数据库说明、网络参数配置、效果对比部分，及代码的上传和管理
- 作者4：李如意 海报的主流方法介绍、海报排版编辑
- 作者5：隋雨辰 海报的任务以及原理部分，海报的整合。

