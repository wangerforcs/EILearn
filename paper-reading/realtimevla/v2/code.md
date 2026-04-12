# Realtime-VLA V2 代码解析

## 1. 核心点

- 服务端时间轴优化: [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L1)
- 客户端执行器与本地控制: [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L1)

## 2. 先看系统执行链路

从 [client/local_client.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/local_client.py#L1) 可以看出，客户端 runtime 大体是多线程结构：

- 状态线程持续读取机器人状态
- 图像线程持续读取相机
- 心跳线程按固定节奏发执行命令
- 推理线程向服务端请求未来动作
- 某些 executor 还会单独开 control thread 做更高频的局部控制

模型推理、观测采集、动作执行三件事被解耦成异步系统。

## 3. `server/optimizer.py` 在做什么

## 3.1 文件定位

[server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L10) 定义了统一优化接口 `BaseOptimizer`，目前有两个实现：

- `PassThroughOptimizer`: 什么都不做，原样返回动作
- `TimeParameterizationMPC`: 对动作序列做时间重参数化

主要是优化speed加速后的动作序列。

## 3.2 `TimeParameterizationMPC` 的核心变量

在 [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L46) 初始化时，几个关键参数是：

- `dt_ref`: 目标参考时间间隔
- `dt_min`, `dt_max`: 每段动作允许的最小/最大时间
- `s_ref = 1 / dt_ref`
- `s_min = 1 / dt_max`
- `s_max = 1 / dt_min`
- `optim_dims`: 只优化哪些维度
- `v_max`: 速度上界
- `H`: 规划 horizon

优化每段的时间倒数 `s`。

## 3.3 `solve_qp()` 的优化目标

核心求解在 [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L73)。

它做了一个 QP，变量是当前窗口内每一段的 `s_i`，目标包含两类项：

- 时间项: 希望 `s_i` 接近 `s_ref`
- 加速度项: 惩罚相邻段推进速度变化太大

对应代码：

- 时间项的二次部分: [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L83)
- 加速度耦合项: [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L86)
- 线性项 `q`: [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L93)

我对这部分的理解是：
作者在空间路径基本固定的情况下，优化“每一小段多快通过”，让整体既尽量快，又避免速度变化太突兀。

## 3.4 约束是什么

QP 的约束也很清楚：

- 基本 box 约束: `s_min <= s_i <= s_max`
- 如果设置了 `v_max`，再额外限制每段速度不超过上限

对应代码在 [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L96) 到 [server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L107)。

这一段很值得记：
真实约束并不是直接写在 waypoint 本身上，而是通过 `||dp_i|| * s_i <= v_max` 转成对时间尺度的约束。

## 3.5 `re_allocate()` 在做什么

[server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L116) 的 `re_allocate()` 会根据新时间轴重新插值，生成按固定 `dt_ref` 采样的等长动作序列。

所以整个优化流程是：

1. 先把模型输出看成 waypoints
2. 计算相邻 waypoint 差分 `dp`
3. 求一组更合理的时间尺度 `s_traj`
4. 转成累计时间 `t`
5. 在新的时间轴上重新插值出动作序列

这就是“时间轴优化”在代码里的具体落点。

## 3.6 `optimize()` 的入口含义

[server/optimizer.py](https://github.com/dexmal/realtime-vla-v2/blob/main/server/optimizer.py#L143) 说明这个优化器的输入输出都还是 action list。

也就是说，从调用方视角看：

- 输入: 模型输出的 future actions
- 输出: 语义尽量不变、但时间分布更合理的 future actions

这是一个非常干净的接口设计，方便后面继续插别的优化器。

## 4. `client/executor.py` 的角色

如果说服务端优化解决的是“未来动作序列怎么排时间”，  
那么客户端 executor 解决的是“这一刻到底该给机器人发什么”。

这个文件很长，但从结构上可以分成四层：

- 通用工具层
- `RawActionExecutor`
- `OnDeviceMpcExecutor`
- 本地 planner 与估计器

## 5. 先看几个关键基础模块

## 5.1 `ReplayEstimator`

[client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L33) 的 `ReplayEstimator` 很关键。

它维护命令历史 `cmd_hist` 和最近一次观测锚点，然后用一阶系统

`x = y_active + (x - y_active) * exp(-dt / tau)`

去重放从锚点到当前时刻的状态演化。

作用是

- 在观测延迟存在时估计“现在的机器人状态”
- 用观测和预测之间的偏差做接触检测

接触判断逻辑在 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L50) 到 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L76)。


## 5.2 `ProgressManager`

[client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L106) 的 `ProgressManager` 很简单，但对 MPC executor 很有用。

它不直接保存“当前执行到第几步”，而是累计一个浮点 `progress`，再把可消耗的整数步数取出来。  
这适合处理“执行推进速度并不完全等于 1 step / tick”的情况。

## 5.3 `AcadosPlanner`

[client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L170) 是本地 MPC planner。

它把每个规划维度建成一个离散一阶系统：

- 状态 `x` 包含当前估计关节位置、上一次命令、上上次命令
- 控制量 `u` 是当前要发送的命令
- 目标是跟踪 AI 给出的未来参考 `r_ref`
- 同时约束位置、速度、一次差分、二次差分

这里的状态设计很实用：

- `x_q`: 当前估计状态
- `y_prev`: 上一次发送的命令
- `y_prev2`: 上上次发送的命令

这样一来，`dy` 和 `ddy` 都能直接写进 cost 和 constraint。

## 6. `AcadosPlanner` 具体怎么约束

在 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L214) 到 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L221)，cost 由这些误差组成：

- 跟踪误差 `x_q - r_ref`
- 命令和参考的偏差 `y_cmd - r_ref`
- 命令和当前状态偏差 `y_cmd - x_q`
- 一次差分 `dy`
- 二次差分 `ddy`

在 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L253) 到 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L264)，又把下面这些量作为硬约束：

- `y_cmd - x_q`
- `y_cmd - y_prev`
- `y_cmd - 2*y_prev + y_prev2`

这对应的直觉是：

- 不允许一步跳太远
- 不允许命令变化太快
- 不允许加速度太尖

这也是它能做到“smooth and accurate”的代码基础。

## 7. 接触模式是怎么处理的

`AcadosPlanner.solve()` 在 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L336) 里有一个 `contact_mode`。

当检测到接触时：

- 参考轨迹不再继续追未来 AI 动作，而是重复当前估计状态
- 约束边界会通过 `contact_v_scale` 收紧

对应代码：

- 切换约束: [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L293)
- 接触时参考冻结: [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L353)

也就是说，接触时系统优先稳住，而不是继续激进跟踪模型输出。

## 8. `RawActionExecutor` 怎么工作

`RawActionExecutor` 从 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L746) 开始。

只做平滑和插值的模式，核心思路是：

- 推理线程把 future actions 放进队列
- 心跳线程按执行节奏取动作
- 控制线程在相邻动作之间插值，并做平滑与前馈修正

### 8.1 推理前如何构造上下文

[client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L1027) 的 `prepare_infer_payload()` 会：

- 根据当前推理延迟估计未来会执行到哪一步
- 如果队列太短，先 pad
- 结合图像时间戳和观测延迟，构造 `state_trajectory`
- 把 `predicted_steps` 作为 `state_delta` 发给服务端


### 8.2 推理回来后如何接上当前执行进度

[client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L1104) 的 `on_infer_actions()` 会根据：

- 请求发出时的执行步数
- 当前已经执行了多少步
- 当时预测的 anchor 索引

决定把新动作覆盖到队列的哪个位置，而不是无脑整个替换。

这非常关键，因为真实系统里推理结果返回时，机器人已经继续往前动了。

### 8.3 心跳线程发什么

[client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L954) 的 `heartbeat_step()` 负责较低频地推进主动作队列。

这里会：

- 出队当前动作
- 估算到下一动作所需最小时间间隔
- 记录 `raw_pre_smooth_action` / `pre_smooth_action` / `post_smooth_action`

可以看到“原始动作、平滑前、平滑后”三条轨迹。

### 8.4 控制线程做什么

[client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L1156) 的 `control_step()` 是更高频的局部平滑层。

它会做：

- 在线插值
- 可选 Savitzky-Golay 平滑
- `ForwardTracker` 前馈跟踪
- 再发给 actuator

这里我会把它理解成一个 lightweight local servo layer。

## 9. `OnDeviceMpcExecutor` 怎么工作

`OnDeviceMpcExecutor` 从 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L1288) 开始，是这个项目更有代表性的执行模式。

它的核心特点是：

- 不是直接发模型动作
- 也不是简单插值
- 而是把模型未来动作当作参考轨迹，再用本地 MPC 去追踪

### 9.1 初始化时建立什么状态

在 [client/executor.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/executor.py#L1363) 的 `__post_init__()` 中，它会建立：

- `AcadosPlanner`
- `ReplayEstimator`
- `ProgressManager`
- future action 队列
- 状态历史
- 紧急暂停相关状态

这说明这个 executor 已经不是单纯“动作队列消费者”，而是一个小型本地控制系统。

### 9.2 为什么它比 raw executor 更闭环

`RawActionExecutor` 的核心是“排队 + 插值 + 平滑”。  
`OnDeviceMpcExecutor` 的核心则是“状态估计 + 参考轨迹 + 优化求解 + 约束跟踪”。

所以如果任务里接触较多、精度要求更高，本地 MPC 方案会更稳。

## 10. `client/local_client.py` 如何把这些拼起来

在 [client/builders.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/builders.py#L40) 到 [client/builders.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/builders.py#L47)，运行时会根据配置选择：

- `raw_action`
- `ondevice_mpc`

而 [client/local_client.py](https://github.com/dexmal/realtime-vla-v2/blob/main/client/local_client.py#L115) 之后的主循环负责：

- 持续采集状态和图像
- 调 executor 构造推理上下文
- 把服务端动作回填进 executor
- 让 heartbeat/control thread 持续真正执行

所以真正的架构是：

`observer -> local runtime -> infer server -> optimizer -> executor -> actuator`

## 11. 对代码的整体理解

1. 模型先产生 future action list
2. 服务端 optimizer 先做时间重参数化
3. 客户端根据延迟预测应该接到未来哪一步
4. raw executor 模式下做插值、平滑、前馈
5. MPC executor 模式下把动作当参考轨迹再解一次本地最优控制
6. 最终再转换成执行器命令