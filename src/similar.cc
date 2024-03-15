#include <paddle/extension.h>

std::vector<paddle::Tensor> similar_cuda_forward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const int kH,
    const int kW);

std::vector<paddle::Tensor> similar_impl_forward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const int kH,
    const int kW) {
        return similar_cuda_forward(x_ori, x_loc, kH, kW);
}

std::vector<paddle::Tensor> similar_forward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const int kH,
    const int kW) {
        return similar_impl_forward(x_ori, x_loc, kH, kW);
}

std::vector<paddle::Tensor> similar_cuda_backward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const paddle::Tensor &grad_out,
    const int kH,
    const int kW);

std::vector<paddle::Tensor> similar_impl_backward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const paddle::Tensor &grad_out,
    const int kH,
    const int kW) {
        return similar_cuda_backward(x_ori, x_loc, grad_out, kH, kW);
}

std::vector<paddle::Tensor> similar_backward(
    const paddle::Tensor &x_ori,
    const paddle::Tensor &x_loc,
    const paddle::Tensor &grad_out,
    const int kH,
    const int kW) {
        return similar_impl_backward(x_ori, x_loc, grad_out, kH, kW);
}

// shape infer
std::vector<std::vector<int64_t>> SimilarInferShape(
    std::vector<int64_t> x_ori_shape,
    std::vector<int64_t> x_loc_shape,
    int kH,
    int kW) {
  return {
      {x_ori_shape[0], x_ori_shape[2], x_ori_shape[3], kH * kW}};
}

// data type infer
std::vector<paddle::DataType> SimilarInferDtype(
    paddle::DataType x_ori_dtype,
    paddle::DataType x_loc_dtype) {
  return {x_ori_dtype};
}

// build forward op
PD_BUILD_OP(similar)
    .Inputs({"x_ori", "x_loc"})
    .Attrs({"kH: int", "kW: int"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(similar_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(SimilarInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SimilarInferDtype));

// build backward op
PD_BUILD_GRAD_OP(similar)
    .Inputs({paddle::Grad("output"), "x_ori", "x_loc"})
    .Attrs({"kH: int", "kW: int"})
    .Outputs({paddle::Grad("x_ori"), paddle::Grad("x_loc")})
    .SetKernelFn(PD_KERNEL(similar_backward));
