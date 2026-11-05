#!/usr/bin/env python3
"""Apply the CANN source patches to a fresh opencv + opencv_contrib checkout.

These patches are what make opencv mainline (5.0.0) compile + link against
CANN 9.1.0 on aarch64. Each is idempotent (checks before patching), so
re-running on an already-patched tree is a no-op.

Run from the WORK dir that contains the two checkouts as siblings:

    cd <WORK>                      # e.g. /home/coder/work
    python3 patch_opencv_cann.py   # expects ./opencv and ./opencv_contrib

Only 3 upstream incompatibilities are worked around; every one of them is
reproduced and justified in docs/Quick-start-Ascend.md §4:

  1. (a-d) OpenCV 5.0 renamed the shape type to cv::MatShape and dropped
     vector<int>::front().  Upstream's CANN backend still uses the 4.x
     spelling, so it does not compile.  Fix = ONE added cv::MatShape
     overload (upstream's vector<int> constructor is left untouched) plus
     the 3 assignments that no longer type-check.
  2. (e-f) CANN's all_ops.h declares ~1500 ops; pulling it into every dnn
     TU (op_cann.hpp -> net_impl.hpp -> everything) pushes libopencv_dnn's
     .text past the aarch64 R_AARCH64_CALL26/JUMP26 +/-128 MiB window and
     the link dies with "relocation truncated to fit".  Fix = narrow op
     headers, per TU.
  3. (g) CANN 9.x rejects a NULL stream (EH0008); upstream's CANN ops
     pass the stream through unchanged and also has the `if (rawStream ==
     nullptr)` branch inverted.  Fix = fall back to the context's default
     stream.
  4. (h) NOT patched any more.  The AscendC kernel_launch template has
     the same NULL-stream shape, but reverting it changes no test
     outcome: the 53 cannops cases still pass, and the only AscendC
     kernel we ship (threshold) faults on device with an upstream UB
     overflow either way, so that path is covered by neither variant.
  5. (i-j) NOT patched any more.  The H2D/D2H host-buffer clamp was
     defensive hardening for a >= 25.5.x driver that was never observed to
     report a padded size (measured acl_size == host_size == 4000); the
     crash it was meant to prevent was a driver/toolkit version mismatch
     (25.5.2 driver x CANN 9.1.0) corrupting the heap at exit, which is
     fixed on the host side, not in this source.
"""
from __future__ import annotations

import pathlib


def patch(rel, old, new, *, expect_present=True, count=1):
    p = pathlib.Path(rel)
    if not p.exists():
        if expect_present:
            print(f'(skip) {rel}: file not present')
        return
    s = p.read_text()
    if new in s:
        print(f'(ok)   {rel}: already patched')
        return
    if old not in s:
        if not expect_present:
            return
        raise SystemExit(f'{rel}: pattern not found')
    p.write_text(s.replace(old, new, count))
    print(f'(patch) {rel}: applied')


def main() -> None:
    # (a) op_cann.hpp: add a cv::MatShape overload declaration.  The upstream
    #     std::vector<int> constructor stays exactly as it is.
    patch('opencv/modules/dnn/src/op_cann.hpp',
          '        CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);',
          '        CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);\n'
          '        CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name);')

    # (b) op_cann.cpp: the MatShape overload forwards to the upstream
    #     std::vector<int> constructor (which keeps upstream's body).
    cp = pathlib.Path('opencv/modules/dnn/src/op_cann.cpp')
    cs = cp.read_text()
    deleg = ('\nCannConstOp::CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name)\n'
             '    : CannConstOp(data, dtype, std::vector<int>(shape.begin(), shape.end()), name) {}\n')
    if deleg.strip() in cs:
        print('(ok)   op_cann.cpp: MatShape forwarding ctor already patched')
    elif 'CannConstOp::CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)\n{\n' in cs:
        anchor = ('op_ = std::make_shared<ge::op::Const>(name);\n'
                  '    op_->set_attr_value(*ge_tensor);\n}\n')
        if anchor not in cs:
            raise SystemExit('op_cann.cpp: anchor not found')
        cp.write_text(cs.replace(anchor, anchor + deleg, 1))
        print('(patch) op_cann.cpp: +1 MatShape forwarding ctor')
    else:
        raise SystemExit('op_cann.cpp: std::vector<int> ctor not found')

    # (c)(d) cv::MatShape has no front() and cannot be assigned from a
    #       std::vector<int> -> fix the 3 assignments that no longer compile.
    patch('opencv/modules/dnn/src/layers/gemm_layer.cpp',
          '            shape_C = std::vector<int>{dim};',
          '            shape_C = cv::MatShape(1, dim);')
    patch('opencv/modules/dnn/src/layers/matmul_layer.cpp',
          '                if (real_ndims_C == 1 && bias_shape.front() != 1) {',
          '                if (real_ndims_C == 1 && bias_shape[0] != 1) {')
    patch('opencv/modules/dnn/src/layers/matmul_layer.cpp',
          '                    bias_shape = std::vector<int>{bias_shape.front()};',
          '                    bias_shape = cv::MatShape(1, bias_shape[0]);')

    # (e) all_ops.h -> array_ops.h (1500-op header blows the aarch64 link)
    old_block = '''#ifdef CANN_VERSION_BELOW_6_3_ALPHA002
    #include "op_proto/built-in/inc/all_ops.h" // ge::Conv2D, ...
#else
    #include "built-in/op_proto/inc/all_ops.h" // ge::Conv2D, ...
#endif'''
    new_block = '''#ifdef CANN_VERSION_BELOW_6_3_ALPHA002
    #include "op_proto/built-in/inc/array_ops.h" // ge::op::Const/Data/Identity/Reshape/Unsqueeze
#else
    #include "built-in/op_proto/inc/array_ops.h" // ge::op::Const/Data/Identity/Reshape/Unsqueeze
#endif'''
    patch('opencv/modules/dnn/src/op_cann.hpp', old_block, new_block)

    # (f) per-layer narrow headers
    INSERTS = {
        'layers/batch_norm_layer.cpp': ['nn_batch_norm_ops'],
        'layers/concat_layer.cpp': ['split_combination_ops'],
        'layers/convolution_layer.cpp': ['nn_calculation_ops'],
        'layers/deconvolution_layer.cpp': ['nn_calculation_ops'],
        'layers/depth_space_ops_layer.cpp': ['transformation_ops'],
        'layers/elementwise_layers.cpp': ['nonlinear_fuc_ops', 'elewise_calculation_ops'],
        'layers/eltwise_layer.cpp': ['elewise_calculation_ops'],
        'layers/flatten_layer.cpp': ['transformation_ops'],
        'layers/fully_connected_layer.cpp': ['matrix_calculation_ops'],
        'layers/gemm_layer.cpp': ['matrix_calculation_ops'],
        'layers/instance_norm_layer.cpp': ['nn_norm_ops'],
        'layers/layer_norm.cpp': ['nn_norm_ops'],
        'layers/lrn_layer.cpp': ['nn_norm_ops'],
        'layers/matmul_layer.cpp': ['matrix_calculation_ops'],
        'layers/nary_eltwise_layers.cpp': ['elewise_calculation_ops'],
        'layers/padding_layer.cpp': ['pad_ops'],
        'layers/permute_layer.cpp': ['transformation_ops'],
        'layers/pooling_layer.cpp': ['nn_pooling_ops'],
        'layers/reduce_layer.cpp': ['reduce_ops'],
        'layers/resize2_layer.cpp': ['image_ops'],
        'layers/resize_layer.cpp': ['image_ops'],
        'layers/slice_layer.cpp': ['split_combination_ops', 'selection_ops'],
        'layers/softmax_layer.cpp': ['nn_norm_ops'],
    }
    ANCHOR = '#include "../op_cann.hpp"'
    for rel, headers in INSERTS.items():
        p = pathlib.Path('opencv/modules/dnn/src') / rel
        if not p.exists():
            print(f'(skip) {rel}: not present')
            continue
        s = p.read_text()
        if 'built-in/op_proto/inc/' in s:
            continue
        if ANCHOR not in s:
            print(f'(skip) {rel}: anchor missing (file may not need patch in this tag)')
            continue
        block = ANCHOR + '\n' + '\n'.join(
            f'#include "built-in/op_proto/inc/{h}.h"' for h in headers)
        p.write_text(s.replace(ANCHOR, block, 1))
        print(f'(patch) {rel}: +{len(headers)} narrow header(s)')

    # (g) OperatorRunner NULL stream fallback
    p = pathlib.Path('opencv_contrib/modules/cannops/src/cann_call.cpp')
    s = p.read_text()
    old_g = '''OperatorRunner& OperatorRunner::run(AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        for (const auto& ptr : holder)
            stream.addTensorHolder(ptr);
    }
    return *this;
}'''
    new_g = '''OperatorRunner& OperatorRunner::run(AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    aclrtStream execStream = rawStream;
    if (execStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtCtxGetCurrentDefaultStream(&execStream));
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, execStream));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));
    if (rawStream != nullptr)
    {
        for (const auto& ptr : holder)
            stream.addTensorHolder(ptr);
    }
    return *this;
}'''
    # 已打过补丁的 clone（重跑/复用旧 build 目录）里 fix 已存在但带注释行，
    # old_g/new_g 锚点都对不上；用 fix 标记判定，避免 SystemExit 中断冷构建。
    if 'aclrtCtxGetCurrentDefaultStream' in s:
        print('(ok)   cann_call.cpp: NULL-stream fallback already present (prior patch), skip')
    else:
        patch('opencv_contrib/modules/cannops/src/cann_call.cpp', old_g, new_g)

    # (h) 已删除：AscendC kernel_launch 的 NULL 流 fallback 回退后测不出差别
    #（53 个 cannops 用例仍全过；唯一的 AscendC kernel threshold 因上游
    #  UB 越界在设备侧崩，两种写法都挂）。详见
    #  docs/Quick-start-Ascend.md §4c 末尾的说明。


if __name__ == '__main__':
    main()
