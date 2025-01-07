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

from __future__ import annotations

from abc import ABC, abstractclassmethod
from pathlib import Path
from typing import ClassVar, NamedTuple

from typing_extensions import Self

from .envs import ENV_SOT_COLLECT_INFO
from .utils import Singleton


def try_import_graphviz():
    try:
        import graphviz

        return graphviz
    except ImportError:
        return None


class InfoCollector(metaclass=Singleton):
    def __init__(self):
        self._info: dict[str, list[StepInfoBase]] = {}

    def attach(self, cls: type[StepInfoBase], *args, **kwargs) -> None:
        if self.need_collect(cls):
            info = cls(*args, **kwargs)
            self.register(info)

    def register(self, info: StepInfoBase) -> None:
        info_class_name = info.__class__.__name__
        self._info.setdefault(info_class_name, [])
        self._info[info_class_name].append(info)

    def need_collect(self, cls: type[StepInfoBase]) -> bool:
        return cls.SHORT_NAME in ENV_SOT_COLLECT_INFO.get()

    def clear(self):
        self._info.clear()

    def print_report(self):
        if self._info:
            print(self.generate_report())

    def generate_report(self) -> str:
        report = ""
        for info_class_name, info_list in self._info.items():
            cls = info_list[0].__class__
            report += f"{info_class_name} ({cls.SHORT_NAME}):\n"
            report += cls.summary(info_list)
            report += "\n"
        return report


class StepInfoBase(ABC):
    SHORT_NAME: ClassVar[str]

    def __init__(self): ...

    @classmethod
    @abstractclassmethod
    def summary(cls, history: list[Self]) -> str: ...


class NewSymbolHitRateInfo(StepInfoBase):
    SHORT_NAME = "new_symbol_hit_rate"

    def __init__(
        self, input_tensor_ids: list[int], output_tensor_ids: list[int]
    ):
        super().__init__()
        self.input_tensor_ids = input_tensor_ids
        self.output_tensor_ids = output_tensor_ids

    @classmethod
    def summary(cls, history: list[Self]) -> str:
        if len(history) == 0:
            return f"No {cls.SHORT_NAME} info"
        if len(history) == 1:
            return "Only one subgraph is generated"
        known_tensor_ids = set()
        hit_count = 0
        all_count = sum([len(info.input_tensor_ids) for info in history[1:]])
        for i, info in enumerate(history):
            for tensor_id in info.input_tensor_ids:
                # Skip the first graph
                if i == 0:
                    continue
                if tensor_id in known_tensor_ids:
                    hit_count += 1
            for tensor_id in info.output_tensor_ids:
                known_tensor_ids.add(tensor_id)
        summary = f"All tensor count: {all_count}, hit count: {hit_count}\n"
        summary += f"Hit rate: {hit_count / all_count:.2f}"
        return summary


class SubGraphRelationInfo(StepInfoBase):
    SHORT_NAME = "subgraph_relation"
    STEP_UNIQUE_ID = 0

    class ConcreteShapeInfo(NamedTuple):
        id: int
        ir_shape: list[int]
        real_shape: list[int]

    def __init__(
        self,
        subgraph_name: str,
        input_shape_infos: list[SubGraphRelationInfo.ConcreteShapeInfo],
        output_shape_infos: list[SubGraphRelationInfo.ConcreteShapeInfo],
        is_first_call: bool,
        graph_size: int,
    ):
        super().__init__()
        self.subgraph_name = subgraph_name
        self.input_shape_infos = input_shape_infos
        self.output_shape_infos = output_shape_infos
        self.is_first_call = is_first_call
        self.graph_size = graph_size

    @classmethod
    def summary(cls, history: list[Self]) -> str:
        # TODO: attach input shape (with dynamic shape info)
        cls.STEP_UNIQUE_ID += 1
        if len(history) == 0:
            return f"No {cls.SHORT_NAME} info"
        if all(not subgraph_info.is_first_call for subgraph_info in history):
            return "All subgraph are not the first call"
        graphviz = try_import_graphviz()
        if graphviz is None:
            return "Please install graphviz to show the subgraph relation"
        dot = graphviz.Digraph()
        shape_infos = [
            shape_info
            for info in history
            for shape_info in info.input_shape_infos + info.output_shape_infos
        ]

        def to_tensor_node_name(
            shape_info: SubGraphRelationInfo.ConcreteShapeInfo,
        ):
            return f"tensor_{shape_info.id}"

        visited_shape = set()
        for shape_info in shape_infos:
            if shape_info.id in visited_shape:
                continue
            visited_shape.add(shape_info.id)
            dot.node(
                to_tensor_node_name(shape_info),
                f"Tensor {shape_info.id} shape={shape_info.real_shape}",
                shape="rect",
            )
        for i, info in enumerate(history):
            subgraph_id = f"subgraph_{i}"
            dot.node(
                subgraph_id,
                f"Subgraph {i} ({info.subgraph_name}, size={info.graph_size})",
                shape='oval',
                fillcolor='cyan' if info.is_first_call else None,
                style='filled' if info.is_first_call else None,
            )
            for shape_info in info.input_shape_infos:
                dot.edge(
                    to_tensor_node_name(shape_info),
                    subgraph_id,
                    label=str(shape_info.ir_shape),
                )
            for shape_info in info.output_shape_infos:
                dot.edge(
                    subgraph_id,
                    to_tensor_node_name(shape_info),
                    label=str(shape_info.ir_shape),
                )

        directory = Path(".") / "subgraph_relation"
        directory.mkdir(exist_ok=True, parents=True)
        filename = f"subgraph_relation_{cls.STEP_UNIQUE_ID}"
        dot.render(directory / filename, format="png", cleanup=True)
        return f"Please check {directory / filename}.png for subgraph relation"
