## [lehome challenge](https://lehome-challenge.com/)

这是我参加的第一个具身智能的比赛，比赛3月初开始，4月底结束，历时两个月。不过我其实在4月多才开始接触到并开始训练(主要还是一个xhs求助帖认识了UCAS做具身的一些同学，然后把我拉进了他们的队伍)。

这个比赛要求训练模型操作lerobot机械臂(action dim=12)，完成lehome中的4个叠衣服的任务，分别是长袖、短袖、长裤、短裤。每个子任务提供了10件衣服，每一件有25条数据，所以一共有4\*10\*25=1000条数据。

![alt text](./assets/image.png)

以下将描述我们的参赛过程和对应的细节。

### 一阶段 classifier + ACT

使用resnet对图片中的garment进行分类，得到对应的类别后再使用ACT模型进行动作预测。为每一种类型的衣物单独在其对应的数据上训练一个ACT模型，输入为当前状态和动作历史，输出为当前动作。

输入输出如下
```yaml
  input_features:
    observation.state:
      type: STATE
      shape: [12]
    observation.images.top_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.left_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.right_rgb:
      type: VISUAL
      shape: [3, 480, 640]
  
  output_features:
    action:
      type: ACTION
      shape: [12]
```

这部分大概就是我加入时队伍的进度，整体上提交后的测评有**42**%的成功率，后续刷分应该提高了一点，但是对比leaderboard上其他队伍的成功率(**60%+**)，还是有不小的差距，所以我基本没有用ACT。


### 二阶段 PI05

这是我主要使用的模型，其他队友也尝试了smolvla/lingbotvla/xvla等模型。

为了方便，我直接用的lehome支持的基于lerobot的pi05来训练，关于是否使用深度图的问题，我决定先不用，因为pi05预训练并没有使用深度信息，所以依然保持和ACT一样的输入输出格式。




### 三阶段 SF PI05

之前在xhs刷到过SF(Spatial Forcing)作者的帖子和评论，这是一种使用辅助任务来提升VLA的3D空间理解能力的方法，感觉可能适合这个任务，因为有不少失败都是因为抓错袖子位置(隔空)或者直接对折不管袖子。这个方法具体是用VLM的中间层特征经过投影和VGGT的3D特征对齐，来提升VLA的空间理解能力。原仓库是基于openpi的，我迁移到了使用lerobot的pi05上，训练配置和之前差不多，主要是增加了一个辅助任务loss。

不过出乎预料的是，训练出来的模型在测试集上的表现反而比之前的pi05更差了。