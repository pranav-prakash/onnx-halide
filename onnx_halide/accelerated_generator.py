from .halide_generator import HalideGraphVisitor, HalideNodeVisitor
from .types import VI
import numpy as np
import os
from os.path import join, dirname
from . import __path__
from math import floor, ceil
from .environment_link import Environment
from .base_generator import BaseNodeVisitor

from onnx.onnx_ml_pb2 import NodeProto, TypeProto
from typing import Dict, List, Set, Tuple, Union
class AcceleratedGraphVisitor(HalideGraphVisitor):
    '''Must manually make a copy since node_lookup is a class var'''
    node_lookup = HalideGraphVisitor.node_lookup.copy()
    pass

class AcceleratedNodeVisitor(BaseNodeVisitor):
    attr_fields = {}

    def __init__(self, **kwargs) -> None:
        BaseNodeVisitor.__init__(self, **kwargs)

    def generate_alg(self):
        return None

    def visit(self, node: NodeProto, value_info: Dict[str, TypeProto]) -> Tuple[List[str], Set[str], Set[str]]:
        BaseNodeVisitor.visit(self, node, value_info)
        alg = self.generate_alg()
        if alg is None:
            '''If we can't handle a given node type, punt to Halide'''
            return HalideGraphVisitor.node_lookup[node.op_type](temp_dir=self.temp_dir).visit(node, value_info)

        import pdb; pdb.set_trace()
        code = []
        objects = set([])
        headers = set([])
        return code, objects, headers

    pass

class AcceleratedMatmulVisitor(AcceleratedNodeVisitor):
    op_type = "MatMul"
    def generate_alg(self):
        return None

AcceleratedGraphVisitor.register(AcceleratedMatmulVisitor)

