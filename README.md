# image_affine
affine image from current frame to target frame
在[BoT-SORT](https://github.com/NirAharon/BoT-SORT)的GMC基础上进行了一点点修改：稍微优化了ECC算法，其中`applyEcc_v2`用图像金字塔来加速计算，再低分辨率时计算初始解，再高分辨率时优化解，实现加速；`applyEcc_v3`是在v2基础上初始解先用ORB特征匹配求出仿射变换初始解，不过效果不是很显著。

<img src="https://github.com/WelY1/image_affine/blob/main/ECC.png" width="600px">

# 如何使用
```python
# 首先定义一个对象，正常1080p的图片使用ORB的话基本可以控制在100ms以内，ECC建议至少缩小2倍。
# method: 'orb', 'ecc', 'OptFlow', 'sift'
# downscale: (should >= 1) default=1
from cmc import GMC
align = GMC(method,downscale)

# 然后直接计算每一帧的仿射变换，第一帧默认作为参考帧
H = align.apply(frame)
```

# ECC加速效果

在一段视频序列(origin.avi)上测试

|下采样倍数|金字塔层数|检测时延ms/frame|与初始帧的平均IOU|
|--|--|--|--|
|1|-|3158|0.693|
|1|2|564.36|0.710|
|1|3|177.67|0.732|
|1|4|106.66|0.760|
|4|-|67.534|0.713|
|4|2|19.791|0.759|

可以看到使用图像金字塔加速后不仅速度极大提升，而且防抖效果也有一点改善。

不过在相机运动剧烈场景下速度和效果都不如ORB，而相机抖动缓慢的场景ECC的效果更好更平滑。
