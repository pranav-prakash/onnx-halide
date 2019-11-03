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
    '''Manually make a copy since node_lookup is a class var'''
    node_lookup = HalideGraphVisitor.node_lookup.copy()
    pass

class AcceleratedNodeVisitor(BaseNodeVisitor):
    attr_fields = {}

    def __init__(self, **kwargs) -> None:
        BaseNodeVisitor.__init__(self, **kwargs)

    def generate_alg(self):
        pass

    def visit(self, node: NodeProto, value_info: Dict[str, TypeProto]) -> Tuple[List[str], Set[str], Set[str]]:
        BaseNodeVisitor.visit(self, node, value_info)
        alg = self.generate_alg()
        if alg is None:
            '''If we can't handle a given node type, punt to Halide'''
            return HalideGraphVisitor.node_lookup[node.op_type]().visit(node, value_info)
        else:
            return alg

    pass

class AcceleratedMatmulVisitor(AcceleratedNodeVisitor):
    op_type = "MatMul"
    def generate_alg(self) -> Tuple[List[str], Set[str], Set[str]]:
        ip0 = VI(self.value_info[self.inputs[0]])
        ip1 = VI(self.value_info[self.inputs[1]])
        op = VI(self.value_info[self.outputs[0]])
        # Right now skip broadcast support
        if ip0.dims == ip1.dims == 2:
            matmul = """
#include <stdlib.h>
#include <stdio.h>

template<typename T>
inline void matmul(int dimI, int dimJ, int dimK, T in1, T in2, T out) {
    printf("Hello from cpp matmul!\\n");
	for (int i = 0; i < dimI; i++) {
		for (int j = 0; j < dimJ; j++) {
			out[i * dimJ + j] = 0;
			for (int k = 0; k < dimK; k++) {
				out[i * dimJ + j] += in1[i * dimK + k] * in2[k * dimJ + j];
			}
            //printf("%f ", out[i * dimJ + j]);
		}
        printf("\\n");
	}
}"""
            hfile = "\"{}\"".format(Environment.create_header_from_src("cppmatmul", matmul))
            code = "matmul({}, {}, {}, {}, {}, {});".format(
                ip0.shape[0], ip1.shape[1], ip0.shape[1], self.inputs[0], self.inputs[1], self.outputs[0])
            return [code], set(), {hfile}

        return None
AcceleratedGraphVisitor.register(AcceleratedMatmulVisitor)

class AcceleratedReluVisitor(AcceleratedNodeVisitor):
    op_type = "Relu"
    def generate_alg(self):
        ip0 = VI(self.value_info[self.inputs[0]])
        relu = """
#include <stdlib.h>
#include <stdio.h>

template<typename T>
inline void relu(int len, T in, T out) {
    printf("Hello from cpp relu!\\n");
	for (int i = 0; i < len; i++) {
		out[i] = in[i] < 0 ? 0 : in[i];
	}
}"""
        hfile = "\"{}\"".format(Environment.create_header_from_src("cpprelu", relu))
        code = "relu({}, {}, {});".format(ip0.size, self.inputs[0], self.outputs[0])    
        return [code], set(), {hfile}
AcceleratedGraphVisitor.register(AcceleratedReluVisitor)




