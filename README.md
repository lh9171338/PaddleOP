[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) PaddleOP
===

这是一个Paddle算子demo，实现了local_attn算子，用于计算[Image Local Attention](https://github.com/zzd1992/Image-Local-Attention)

# 安装

```shell
git clone ssh://lihao57@icode.baidu.com:8235/baidu/personal-code/PaddleOP

# 编译算子
cd PaddleOP
python setup.py install
```

# 测试

```shell
python test.py

# print信息
# Tensor(shape=[1, 1, 3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[0.68500221, 0.93662095, 0.94141233],
#           [0.99092519, 1.        , 0.99876219],
#           [0.99886143, 0.99983227, 0.99984574]]]])
# Tensor(shape=[1, 3, 3, 9], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[0., 0., 0., 0., 1., 1., 0., 1., 1.],
#           [0., 0., 0., 2., 2., 2., 2., 2., 2.],
#           [0., 0., 0., 3., 3., 0., 3., 3., 0.]],

#          [[0., 4., 4., 0., 4., 4., 0., 4., 4.],
#           [5., 5., 5., 5., 5., 5., 5., 5., 5.],
#           [6., 6., 0., 6., 6., 0., 6., 6., 0.]],

#          [[0., 7., 7., 0., 7., 7., 0., 0., 0.],
#           [8., 8., 8., 8., 8., 8., 0., 0., 0.],
#           [9., 9., 0., 9., 9., 0., 0., 0., 0.]]]])
```
