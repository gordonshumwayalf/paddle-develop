# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np
from auto_parallel_pass_test_base_deprecated import AutoParallelPassTestBase

import paddle
from paddle.distributed import fleet


class TestPF16Pass(AutoParallelPassTestBase):
    def init(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-5
        self.atol = 1e-8

        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)

    def apply_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_white_list": [
                'softmax',
                'layer_norm',
                'gelu',
            ],
            "custom_black_list": [
                'c_softmax_with_cross_entropy',
                'elementwise_div',
                'reduce_sum',
            ],
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
            "use_pure_fp16": True,
            "use_fp16_guard": False,
        }
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_bs_8(self):
        self.check_main(
            gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000
        )

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        return self.get_gpt_model(
            "mp", place, batch_size, sequence_len, vocab_size
        )


if __name__ == "__main__":
    unittest.main()
