from typing import List, Dict, Optional
import itertools
import os
import json
from collections import namedtuple

import torch
from torch import nn
import numpy as np

from pruning.utils import cast_tuple_to_scalar


OpConfig = namedtuple(
    "OpConfig", [
        "type", "cin", "cout", "height", "width",
        "ksize", "stride", "pad", "dilation"
    ])


class PrunedNetwork(nn.Module):
    _PREDEFINED_OPS_PREFIX = "predefined_ops:"

    def __init__(
        self, network: nn.Module,
        pruning_config: dict,
    ):
        super().__init__()
        self.network = network
        self.predefined_ops = {}
        for dic in pruning_config["predefined_ops"]:
            name = dic["name"]
            assert name not in self.predefined_ops
            self.predefined_ops[name] = dic

        self.dependencies = pruning_config["dependencies"]
        self.effects = pruning_config["effects"]
        self.input_shapes = pruning_config["input_shapes"]

    def set_pruning_state(self, pruning_state: Dict[str, int]):
        dic = self.get_pruning_state()
        for pruning_point, channels in pruning_state.items():
            self.prune(pruning_point, list(range(
                dic[pruning_point] - channels
            )))

    def get_pruning_state(self) -> Dict[str, int]:
        """Get pruning state
        Returns:
            a dict which maps operator name (pruning point) to remaining output channels
        """
        pruning_state = {}
        for pruning_point, dic in self.dependencies.items():
            cout = self.get_model_attr_by_name(pruning_point).weight.shape[0]
            for conv in dic["Conv"]:
                new_cout = self.get_model_attr_by_name(conv).weight.shape[0]
                assert cout == new_cout
            for bn in dic["BatchNormalization"]:
                new_cout = self.get_model_attr_by_name(bn).weight.shape[0]
                assert cout == new_cout
            pruning_state[pruning_point] = cout

        return pruning_state

    def load_pruning_state(self, path: str):
        with open(path, "r") as f:
            pruning_state = json.load(f)
            self.set_pruning_state(pruning_state)

    def save_pruning_state(self, path: str):
        with open(path, "w") as f:
            json.dump(self.get_pruning_state(), f)

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    @property
    def __class__(self):
        return type(self.network)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super().state_dict(destination, "", keep_vars)
        keys = list(ret.keys())
        for key in keys:
            if key.startswith("network."):
                new_key = key[len("network."):]
            else:
                new_key = key
            new_key = prefix + new_key
            ret[new_key] = ret[key]
            del ret[key]
        return ret

    @classmethod
    def _reconstruct_op(cls, module: nn.Module, **kwargs):
        assert type(module) in [nn.Conv2d, nn.BatchNorm2d]
        if isinstance(module, nn.Conv2d):
            new_op_kwargs = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "groups": module.groups,
                "bias": module.bias is not None,
                "padding_mode": module.padding_mode,
                **kwargs
            }
            return nn.Conv2d(**new_op_kwargs)
        elif isinstance(module, nn.BatchNorm2d):
            new_op_kwargs = {
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
                "affine": module.affine,
                "track_running_stats": module.track_running_stats,
                **kwargs
            }
            return nn.BatchNorm2d(**new_op_kwargs)

    @classmethod
    def _copy_module_weight(
        cls, to_module: nn.Module, from_module: nn.Module,
        pruned_channel_idxs: List[int],
        channel_axis: int,
    ):
        assert type(from_module) == type(to_module)
        assert type(from_module) in [nn.Conv2d, nn.BatchNorm2d]
        assert channel_axis in [0, 1]
        assert channel_axis == 0 or isinstance(from_module, nn.Conv2d)

        if isinstance(from_module, nn.Conv2d):
            weight_names = {
                "weight": True,
                "bias": channel_axis == 0
            }
        elif isinstance(from_module, nn.BatchNorm2d):
            weight_names = {
                "weight": True, "bias": True,
                "running_mean": True, "running_var": True,
                "num_batches_tracked": False,
            }

        for name, need_prune in weight_names.items():
            value = getattr(from_module, name, None)
            if value is None:
                continue
            if need_prune:
                value = np.delete(
                    value.data.numpy(),
                    pruned_channel_idxs,
                    axis=channel_axis
                )
            else:
                value = value.data.numpy()
            getattr(to_module, name).data = torch.from_numpy(value)

    def prune(self, pruning_point: str, channel_idxs: List[int]):
        pruning_point_op_config = self.get_op_config(pruning_point)
        target_num_channels = pruning_point_op_config.cout - len(channel_idxs)

        for conv_name in itertools.chain(self.dependencies[pruning_point]["Conv"], [pruning_point]):
            self._prune_conv(
                conv_name, None,
                target_num_channels, channel_idxs
            )

        for bn_name in self.dependencies[pruning_point]["BatchNormalization"]:
            bn = self.get_model_attr_by_name(bn_name)
            new_bn = self._reconstruct_op(bn, num_features=target_num_channels)
            self._copy_module_weight(new_bn, bn, channel_idxs, 0)
            self.set_model_attr_by_name(bn_name, new_bn)

        for pruned_conv_name in itertools.chain(self.dependencies[pruning_point]["Conv"], [pruning_point]):
            for conv_name in self.effects.get(pruned_conv_name, []):
                op_config = self.get_op_config(conv_name)
                if op_config.type == "MaxPool":
                    continue
                self._prune_conv(
                    conv_name, target_num_channels,
                    None, channel_idxs
                )

    def _prune_conv(self, conv_name, cin: Optional[int], cout: Optional[int], channel_idxs: List[int]):
        assert cin is not None or cout is not None
        conv_op_config = self.get_op_config(conv_name)
        assert conv_op_config.type == "Conv"
        conditions = [
            cin is None or conv_op_config.cin == cin,
            cout is None or conv_op_config.cout == cout
        ]
        assert conditions[0] or conditions[1]
        if conditions[0] and conditions[1]:
            # already pruned
            return

        if conv_name.startswith(self._PREDEFINED_OPS_PREFIX):
            conv = self._fetch_predefined_op(conv_name)
            weight = self.get_model_attr_by_name(conv["weight"])
            bias = self.get_model_attr_by_name(conv["bias"])
            if not conditions[1]:
                # prune cout
                new_weight = np.delete(
                    weight.data.numpy(), channel_idxs, axis=0)
                new_bias = np.delete(bias.data.numpy(), channel_idxs, axis=0)
                weight.data = torch.from_numpy(new_weight)
                bias.data = torch.from_numpy(new_bias)
            elif not conditions[0]:
                # prune cin
                new_weight = np.delete(
                    weight.data.numpy(), channel_idxs, axis=1)
                weight.data = torch.from_numpy(new_weight)
        else:
            conv = self.get_model_attr_by_name(conv_name)
            kwargs = {
                **({"in_channels": cin} if not conditions[0] else {}),
                **({"out_channels": cout} if not conditions[1] else {})
            }
            new_conv = self._reconstruct_op(conv, **kwargs)
            self._copy_module_weight(
                new_conv, conv, channel_idxs, int(conditions[1]))
            self.set_model_attr_by_name(conv_name, new_conv)

    def get_model_attr_by_name(self, name: str) -> nn.Module:
        parts = name.split('.')
        for i in range(len(parts)):
            try:
                idx = int(parts[i])
                parts[i] = idx
            except:
                ...
        node = self.network
        for part in parts:
            if isinstance(part, int):
                node = node[part]
            else:
                node = getattr(node, part)
        return node

    def set_model_attr_by_name(self, name: str, module: nn.Module):
        path, ext = os.path.splitext(name)
        ext = ext[1:]
        try:
            idx = int(ext)
            ext = idx
        except:
            ...
        father = self.get_model_attr_by_name(path)
        if isinstance(ext, int):
            father[ext] = module
        else:
            setattr(father, ext, module)

    def _fetch_predefined_op(self, op_name: str):
        assert op_name.startswith(self._PREDEFINED_OPS_PREFIX)
        op_name = op_name[len(self._PREDEFINED_OPS_PREFIX):]
        return self.predefined_ops[op_name]

    def get_op_config(self, name) -> OpConfig:
        input_shape = self.input_shapes[name]
        common_dic = {
            "height": input_shape[0],
            "width": input_shape[1]
        }

        if name.startswith(self._PREDEFINED_OPS_PREFIX):
            op = self._fetch_predefined_op(name)
            common_dic.update({
                key: value for key, value in op.items()
                if key in ["type", "ksize", "stride", "pad", "dilation"]
            })

            assert op["type"] in ["Conv", "MaxPool"]
            if op["type"] == "Conv":
                weight = self.get_model_attr_by_name(op["weight"])
                return OpConfig(
                    cin=weight.shape[1],
                    cout=weight.shape[0],
                    **common_dic
                )
            elif op["type"] == "MaxPool":
                reference = self.get_op_config(
                    op["num_channels_inference_from"])
                return OpConfig(
                    cin=reference.cout,
                    cout=reference.cout,
                    **common_dic
                )

        else:
            layer = self.get_model_attr_by_name(name)
            assert isinstance(layer, nn.Conv2d)
            common_dic.update({
                "ksize": cast_tuple_to_scalar(layer.kernel_size),
                "stride": cast_tuple_to_scalar(layer.stride),
                "pad": cast_tuple_to_scalar(layer.padding),
                "dilation": cast_tuple_to_scalar(layer.dilation),
            })

            return OpConfig(
                type="Conv",
                cin=layer.in_channels,
                cout=layer.out_channels,
                **common_dic
            )

    def _l2_norm(self, pruning_point: str, channel_idx: int):
        metric = 0
        tot = 0
        for conv_name in itertools.chain(self.dependencies[pruning_point]["Conv"], [pruning_point]):
            weight = self.get_model_attr_by_name(conv_name).weight.data.numpy()
            metric = np.average(weight[channel_idx] ** 2) ** 0.5
            tot += 1
        metric /= tot
        return metric

    def select_channel_idxs(self, pruning_point: str, num_channels: int) -> List[int]:
        cout = self.get_model_attr_by_name(pruning_point).out_channels
        metrics = [None for _ in range(cout)]
        for i in range(cout):
            metrics[i] = (self._l2_norm(pruning_point, i), i)
        metrics = sorted(metrics)
        return list(map(lambda x: x[1], metrics))[:num_channels]
