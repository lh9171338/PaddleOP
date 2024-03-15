import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from local_attn import similar, weighting


class LocalAttention(nn.Layer):
    """
    Local Attention
    """

    def __init__(self, kH, kW):
        super().__init__()

        self.kH = kH
        self.kW = kW

    def forward(self, x1, x2):
        weight = similar(x1, x2, self.kH, self.kW)
        weight_out = weight
        weight = F.softmax(weight, axis=-1)
        out = weighting(x2, weight, self.kH, self.kW)

        return out, weight_out


if __name__ == "__main__":
    q = paddle.to_tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.float32
    )[None, None]
    k = paddle.ones((1, 1, 3, 3), dtype=paddle.float32)
    kH = 3
    kW = 3
    model = LocalAttention(3, 3)
    out, weight_out = model(q, k)
    print(out)
    print(weight_out)
