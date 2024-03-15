#include "paddle/extension.h"
#include "kernels.cuh"


std::vector<paddle::Tensor> similar_cuda_forward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const int kH,
    const int kW
) {
    TypeCheck(x_ori);
    TypeCheck(x_loc);
    const int batch = x_ori.shape()[0];
    const int channels = x_ori.shape()[1];
    const int height = x_ori.shape()[2];
    const int width = x_ori.shape()[3];

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = height * width * patch;
    auto output = paddle::zeros({batch, height, width, patch}, x_ori.type(), paddle::GPUPlace());

    int start_inp = 0, start_out = 0;
    for (int i = 0; i < batch; ++i) {
        f_cc2k<float, double>(
                x_ori.stream(),
                x_ori.data<float>() + start_inp,
                x_loc.data<float>() + start_inp,
                kH, kW, rH, rW,
                patch, channels, height, width,
                per_channel,
                output.data<float>() + start_out
        );
        start_inp += per_input;
        start_out += per_output;
    }

    return {output};
}

//////////////////////////////////////////////////////////////

std::vector<paddle::Tensor> similar_cuda_backward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const paddle::Tensor &grad_out,
    const int kH,
    const int kW
) {
    TypeCheck(x_ori);
    TypeCheck(x_loc);
    const int batch = x_ori.shape()[0];
    const int channels = x_ori.shape()[1];
    const int height = x_ori.shape()[2];
    const int width = x_ori.shape()[3];

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;

    auto grad_x_ori = paddle::zeros({batch, channels, height, width}, x_loc.type(), paddle::GPUPlace());
    auto grad_x_loc = paddle::zeros({batch, channels, height, width}, x_ori.type(), paddle::GPUPlace());

    int start_inp = 0;
    for (int i = 0; i < batch; ++i) {
        auto grad_out_row = grad_out.slice(i, i + 1);
        f_ck2c_ori<float, double>(
                x_loc.stream(),
                x_loc.data<float>() + start_inp,
                grad_out_row.data<float>(),
                kH, kW, rH, rW,
                patch, channels,
                height, width,
                per_channel, per_input,
                grad_x_ori.data<float>() + start_inp
        );
        f_ck2c_loc<float, double>(
                x_ori.stream(),
                x_ori.data<float>() + start_inp,
                grad_out_row.data<float>(),
                kH, kW, rH, rW,
                patch, channels,
                height, width,
                per_channel, per_input,
                grad_x_loc.data<float>() + start_inp
        );
        start_inp += per_input;
    }
    return {grad_x_ori, grad_x_loc};
}
