#!/usr/bin/env python3
"""Apply the CANN source patches to a fresh opencv checkout.

These patches are what make opencv mainline (5.0.0) compile + link against
CANN 9.1.0 on aarch64. Each is idempotent (checks before patching), so
re-running on an already-patched tree is a no-op.

Run from the WORK dir that contains the checkout:

    cd <WORK>                      # e.g. /home/coder/work
    python3 patch_opencv_cann.py   # expects ./opencv

Only 2 upstream incompatibilities are worked around; every one of them is
reproduced and justified in docs/Quick-start-Ascend.md §4:

  1. (a-d) OpenCV 5.0 renamed the shape type to cv::MatShape and dropped
     vector<int>::front().  Upstream's CANN backend still uses the 4.x
     spelling, so it does not compile.  Fix = the upstream signature is
     REPLACED by `const cv::MatShape&` (not extended with a second
     overload), the 15 shape locals that feed it become cv::MatShape, and
     the 3 assignments that no longer type-check are fixed.  Same form as
     the quick-start docs' inline patch blocks -- keeping one form across
     the repo is the point: when upstream adopts the same change, the whole
     group reverts 1:1 instead of leaving a dead vector<int> overload behind.
  2. (e-f) CANN's all_ops.h declares ~1500 ops; pulling it into every dnn
     TU (op_cann.hpp -> net_impl.hpp -> everything) pushes libopencv_dnn's
     .text past the aarch64 R_AARCH64_CALL26/JUMP26 +/-128 MiB window and
     the link dies with "relocation truncated to fit".  Fix = narrow op
     headers, per TU.
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
    # (a) op_cann.hpp: REPLACE the upstream shape parameter with cv::MatShape.
    #     No forwarding overload is kept -- see the module docstring.
    patch('opencv/modules/dnn/src/op_cann.hpp',
          'CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);',
          'CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name);')

    # (b) op_cann.cpp: the matching definition.  Upstream's body is untouched
    #     -- it only iterates shape.begin()/shape.end().
    patch('opencv/modules/dnn/src/op_cann.cpp',
          'CannConstOp::CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)',
          'CannConstOp::CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name)')

    # (c) the 15 shape locals that feed CannConstOp become cv::MatShape:
    #     MatShape's vector<int> constructor is explicit, so a std::vector<int>
    #     does not implicitly convert.
    SHAPE_LOCALS = [
        ('layers/batch_norm_layer.cpp',      'std::vector<int> shape_{channel};'),
        ('layers/elementwise_layers.cpp',    'std::vector<int> shape_{1};'),
        ('layers/elementwise_layers.cpp',    'std::vector<int> shape_{scale.size[0]};'),
        ('layers/slice_layer.cpp',           'std::vector<int> shape_size_splits{(int)size_splits.size()};'),
        ('layers/slice_layer.cpp',           'std::vector<int> split_dim_shape{1};'),
        ('layers/slice_layer.cpp',           'std::vector<int> shape_{dims};'),
        ('layers/padding_layer.cpp',         'std::vector<int> pads_shape{(int)pads.size()};'),
        ('layers/padding_layer.cpp',         'std::vector<int> constant_values_shape{1};'),
        ('layers/reduce_layer.cpp',          'std::vector<int> axes_shape{(int)axes.size()};'),
        ('layers/resize_layer.cpp',          'std::vector<int> shape_of_size_mat{2};'),
        ('layers/resize2_layer.cpp',         'std::vector<int> shape_of_size_mat{2};'),
        ('layers/reshape_layer.cpp',         'std::vector<int> shape_of_shape{(int)newShapeDesc.size()};'),
        ('layers/convolution_layer.cpp',     'std::vector<int> bias_shape{out_channel};'),
        ('layers/deconvolution_layer.cpp',   'std::vector<int> shape_b{out_channel};'),
        ('layers/fully_connected_layer.cpp', 'std::vector<int> bias_shape{weightsMat.size[0]};'),
    ]
    for rel, old in SHAPE_LOCALS:
        patch(f'opencv/modules/dnn/src/{rel}', old, old.replace('std::vector<int> ', 'cv::MatShape '))

    # (d) cv::MatShape has no front() and cannot be assigned from a
    #     std::vector<int> -> fix the 3 assignments that no longer compile.
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


if __name__ == '__main__':
    main()
