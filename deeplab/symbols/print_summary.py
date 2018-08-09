# --------------------------------------------------------
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Copyright (c) 2017 ShanghaiTech PLUS Group
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# Written by Songyang Zhang 
# E-main: sy.zhangbuaa#gmail.com
# --------------------------------------------------------

import json
from mxnet import Symbol
def print_summary(symbol, shape=None, line_length=120, positions=[.44, .64, .74, 1.]):
    """Convert symbol for detail information.

    Parameters
    ----------
    symbol: Symbol
        Symbol to be visualized.
    shape: dict
        A dict of shapes, str->shape (tuple), given input shapes.
    line_length: int
        Rotal length of printed lines
    positions: list
        Relative or absolute positions of log elements in each line.
    Returns
    ------
    None
    """
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    show_shape = False
    if shape is not None:
        show_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes is None:
            raise ValueError("Input shape is incomplete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    heads = set(conf["heads"][0])
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Previous Layer']
    def print_row(fields, positions):
        """Print format row.

        Parameters
        ----------
        fields: list
            Information field.
        positions: list
            Field length ratio.
        Returns
        ------
        None
        """
        line = ''
        for i, field in enumerate(fields):
            line += str(field)
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)
    print('_' * line_length)
    print_row(to_display, positions)
    print('=' * line_length)
    def print_layer_summary(node, out_shape):
        """print layer information

        Parameters
        ----------
        node: dict
            Node information.
        out_shape: dict
            Node shape information.
        Returns
        ------
            Node total parameters.
        """
        op = node["op"]
        pre_node = []
        pre_filter = 0
        if op != "null":
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_node["op"] != "null" or item[0] in heads:
                    # add precede
                    pre_node.append(input_name)
                    if show_shape:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                        else:
                            key = input_name
                        if key in shape_dict:
                            shape = shape_dict[key][1:]
                            pre_filter = pre_filter + int(shape[0])
        cur_param = 0
        if op == 'Convolution':
            if ("no_bias" in node["attrs"]) and (isinstance(node["attrs"]["no_bias"],(bool,int)) and int(node["attrs"]["no_bias"])) or ((isinstance(node["attrs"]["no_bias"],(str)) and bool(node["attrs"]["no_bias"]))):
                cur_param = pre_filter * int(node["attrs"]["num_filter"])
                for k in _str2tuple(node["attrs"]["kernel"]):
                    cur_param *= int(k)
            else:
                cur_param = pre_filter * int(node["attrs"]["num_filter"])
                for k in _str2tuple(node["attrs"]["kernel"]):
                    cur_param *= int(k)
                cur_param += int(node["attrs"]["num_filter"])
        elif op == 'FullyConnected':
            if ("no_bias" in node["attrs"]) and int(node["attrs"]["no_bias"]):
                cur_param = pre_filter * (int(node["attrs"]["num_hidden"]))
            else:
                cur_param = (pre_filter+1) * (int(node["attrs"]["num_hidden"]))
        elif op == 'BatchNorm':
            key = node["name"] + "_output"
            if show_shape:
                num_filter = shape_dict[key][1]
                cur_param = int(num_filter) * 2
        if not pre_node:
            first_connection = ''
        else:
            first_connection = pre_node[0]
        fields = [node['name'] + '(' + op + ')',
                  "x".join([str(x) for x in out_shape]),
                  cur_param,
                  first_connection]
        print_row(fields, positions)
        if len(pre_node) > 1:
            for i in range(1, len(pre_node)):
                fields = ['', '', '', pre_node[i]]
                print_row(fields, positions)
        return cur_param
    total_params = 0
    for i, node in enumerate(nodes):
        out_shape = []
        op = node["op"]
        if op == "null" and i > 0:
            continue
        if op != "null" or i in heads:
            if show_shape:
                if op != "null":
                    key = node["name"] + "_output"
                else:
                    key = node["name"]
                if key in shape_dict:
                    out_shape = shape_dict[key][1:]
        total_params += print_layer_summary(nodes[i], out_shape)
        if i == len(nodes) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)
    print('Total params: %s' % total_params)
    print('_' * line_length)