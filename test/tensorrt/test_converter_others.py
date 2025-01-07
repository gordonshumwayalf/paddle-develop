# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

import paddle
from paddle import _C_ops


def api_wrapper(x):
    return paddle._C_ops.share_data(x)


def multiclass_nms3(
    bboxes,
    scores,
    rois_num=None,
    score_threshold=0.3,
    nms_top_k=4,
    keep_top_k=1,
    nms_threshold=0.3,
    normalized=True,
    nms_eta=1.5,
    background_label=-1,
    return_index=False,
    return_rois_num=True,
    name=None,
):
    attrs = (
        score_threshold,
        nms_top_k,
        keep_top_k,
        nms_threshold,
        normalized,
        nms_eta,
        background_label,
    )
    output, index, nms_rois_num = _C_ops.multiclass_nms3(
        bboxes, scores, rois_num, *attrs
    )
    if not return_index:
        index = None
    return output, nms_rois_num, index


class TestMulticlassNMS3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = multiclass_nms3
        self.api_args = {
            "bboxes": np.random.randn(2, 5, 4).astype("float32"),
            "scores": np.random.randn(2, 4, 5).astype("float32"),
        }
        self.program_config = {"feed_list": ["bboxes", "scores"]}
        self.min_shape = {"bboxes": [1, 5, 4], "scores": [1, 4, 5]}
        self.max_shape = {"bboxes": [3, 5, 4], "scores": [3, 4, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMulticlassNMS3Marker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = multiclass_nms3
        self.api_args = {
            "bboxes": np.random.randn(2, 5, 4, 1).astype("float32"),
            "scores": np.random.randn(2, 4, 5, 1).astype("float32"),
        }
        self.program_config = {"feed_list": ["bboxes", "scores"]}
        self.target_marker_op = "pd_op.multiclass_nms3"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


def set_value(
    x, starts, ends, steps, axes, decrease_axes, none_axes, shape, values
):
    output = _C_ops.set_value(
        x,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
        values,
    )
    return output


def set_value_(
    x, starts, ends, steps, axes, decrease_axes, none_axes, shape, values
):
    output = _C_ops.set_value_(
        x,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
        values,
    )
    return output


def set_value_with_tensor(
    x, values, starts, ends, steps, axes, decrease_axes, none_axes, shape
):
    output = _C_ops.set_value_with_tensor(
        x,
        values,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
    )
    return output


def set_value_with_tensor_(
    x, values, starts, ends, steps, axes, decrease_axes, none_axes, shape
):
    output = _C_ops.set_value_with_tensor_(
        x,
        values,
        starts,
        ends,
        steps,
        axes,
        decrease_axes,
        none_axes,
        shape,
    )
    return output


class TestSetValueTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_trt_result()


# starts/ends/steps is not one element
class TestSetValueMarkerCase1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0, 0],
            "ends": [1, 1],
            "steps": [1, 1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# decrease_axes has element
class TestSetValueMarkerCase2(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [1],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# values has more than one element
class TestSetValueMarkerCase3(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0, 0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# values has int element
class TestSetValueMarkerCase4(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


# starts is not constant value
class TestSetValueMarkerCase5(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": np.zeros([1]).astype("int64"),
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x", "starts"]}
        self.min_shape = {"x": [1, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestSetValue_TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_
        self.api_args = {
            "x": np.ones([10, 2]).astype("float32"),
            "starts": [0],
            "ends": [1],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
            "values": [10.0],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.max_shape = {"x": [20, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSetValueWithTensorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_with_tensor
        self.api_args = {
            "x": np.ones([2, 3, 3]).astype("float32"),
            "values": np.random.randn(2, 2, 3).astype("float32"),
            "starts": [0],
            "ends": [2],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
        }
        self.program_config = {"feed_list": ["x", "values"]}
        self.min_shape = {"x": [1, 3, 3], "values": [1, 2, 3]}
        self.max_shape = {"x": [4, 3, 3], "values": [4, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


# values is int type
class TestSetValueWithTensorMarkerCase1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_with_tensor
        self.api_args = {
            "x": np.ones([2, 3, 3]).astype("float32"),
            "values": np.random.randn(2, 2, 3).astype("int32"),
            "starts": [0],
            "ends": [2],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
        }
        self.program_config = {"feed_list": ["x", "values"]}
        self.min_shape = {"x": [1, 3, 3], "values": [1, 2, 3]}
        self.max_shape = {"x": [4, 3, 3], "values": [4, 2, 3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestSetValueWithTensor_TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = set_value_with_tensor_
        self.api_args = {
            "x": np.ones([2, 3, 3]).astype("float32"),
            "values": np.random.randn(2, 2, 3).astype("float32"),
            "starts": [0],
            "ends": [2],
            "steps": [1],
            "axes": [1],
            "decrease_axes": [],
            "none_axes": [],
            "shape": [],
        }
        self.program_config = {"feed_list": ["x", "values"]}
        self.min_shape = {"x": [1, 3, 3], "values": [1, 2, 3]}
        self.max_shape = {"x": [4, 3, 3], "values": [4, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestShareDataTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = api_wrapper
        self.api_args = {
            "x": np.random.rand(4, 3, 5).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [4, 3, 5]}
        self.max_shape = {"x": [6, 3, 5]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
