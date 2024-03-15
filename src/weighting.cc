#include <paddle/extension.h>

std::vector<paddle::Tensor> weighting_cuda_forward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_weight,
    const int kH,
    const int kW);

std::vector<paddle::Tensor> weighting_impl_forward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_weight,
    const int kH,
    const int kW) {
        return weighting_cuda_forward(x_ori, x_weight, kH, kW);
}

std::vector<paddle::Tensor> weighting_forward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_weight,
    const int kH,
    const int kW) {
        return weighting_impl_forward(x_ori, x_weight, kH, kW);
}

std::vector<paddle::Tensor> weighting_cuda_backward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_weight,
    const paddle::Tensor &grad_out,
    const int kH,
    const int kW);

std::vector<paddle::Tensor> weighting_impl_backward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_weight,
    const paddle::Tensor &grad_out,
    const int kH,
    const int kW) {
        return weighting_cuda_backward(x_ori, x_weight, grad_out, kH, kW);
}

std::vector<paddle::Tensor> weighting_backward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_weight,
    const paddle::Tensor &grad_out,
    const int kH,
    const int kW) {
        return weighting_impl_backward(x_ori, x_weight, grad_out, kH, kW);
}

// shape infer
std::vector<std::vector<int64_t>> WeightingInferShape(
    std::vector<int64_t> x_ori_shape,
    std::vector<int64_t> x_weight_shape) {
  return {x_ori_shape};
}

// data type infer
std::vector<paddle::DataType> WeightingInferDtype(
    paddle::DataType x_ori_dtype,
    paddle::DataType x_weight_dtype) {
  return {x_ori_dtype};
}

// build forward op
PD_BUILD_OP(weighting)
    .Inputs({"x_ori", "x_weight"})
    .Attrs({"kH: int", "kW: int"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(weighting_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightingInferDtype));

// build backward op
PD_BUILD_GRAD_OP(weighting)
    .Inputs({paddle::Grad("output"), "x_ori", "x_weight"})
    .Attrs({"kH: int", "kW: int"})
    .Outputs({paddle::Grad("x_ori"), paddle::Grad("x_weight")})
    .SetKernelFn(PD_KERNEL(weighting_backward));
