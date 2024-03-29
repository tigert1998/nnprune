import os
import os.path as osp
import json
from copy import deepcopy
from typing import Optional
import pickle
from tempfile import mkstemp
import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from nnprune.pruned_network import PrunedNetwork


class NetadaptPruner:
    """NetadaptPruner

    Args:
        unpruned_model: unpruned model
        pruning_config_path: pruning_config path for unpruned_model
        pruning_plan: A dict that defines the pruning process. Available keys:
            step: the alignment of channels for the pruned model;
            delta_resource: the least resource reduction in a round of pruning;
            resource_budget: resource budget;
            delta_resource_decay: delta_resource decay rate. Default: 1 (no decay)
        work_dir: the work directory to save log and checkpoints
    """

    def __init__(
        self,
        unpruned_model: nn.Module,
        pruning_config_path: str,
        pruning_plan: dict,
        work_dir: str,
    ):
        self.pruning_config_path = pruning_config_path
        with open(self.pruning_config_path, "r") as f:
            self.pruning_config = json.load(f)

        self.model = PrunedNetwork(unpruned_model, self.pruning_config)
        self.pruning_points = list(self.model.dependencies.keys())

        self.pruning_plan = pruning_plan
        self.work_dir = work_dir

        self.num_gpus = torch.cuda.device_count()
        self.rank, self.work_size = dist.get_rank(), dist.get_world_size()

        self.logger = self.get_logger()
        self.iteration = None

    def construct_model(self) -> nn.Module:
        raise NotImplementedError()

    def evaluate_resource(self, model: PrunedNetwork) -> float:
        raise NotImplementedError()

    def evaluate_accuracy(self, model: PrunedNetwork) -> float:
        raise NotImplementedError()

    def finetune(self, model: PrunedNetwork, term: str):
        raise NotImplementedError()

    def clean_tmp_files(self):
        raise NotImplementedError()

    def get_logger(self):
        raise NotImplementedError()

    def _log_with_iter(self, msg, level=logging.INFO):
        if self.iteration is None:
            iteration_str = "[iteration unknown]"
        else:
            iteration_str = "[iteration #{}]".format(self.iteration)
        self.logger.log(level, "{} {}".format(iteration_str, msg))

    def run(self):
        delta_resource = self.pruning_plan["delta_resource"]
        resource_budget = self.pruning_plan["resource_budget"]
        delta_resource_decay = self.pruning_plan.get("delta_resource_decay", 1)

        # resume progress
        self.iteration = 0
        while True:
            ckpt_name = self._get_ckpt_name(self.iteration)
            if not osp.exists(osp.join(self.work_dir, ckpt_name + ".pth")):
                if self.iteration >= 1:
                    resource = self._load_model(
                        self._get_ckpt_name(self.iteration - 1))["resource"]
                else:
                    resource = self.evaluate_resource(self.model)
                break
            self.iteration += 1

        self.logger.info(
            "resume pruning from iteration #{}".format(self.iteration))

        while resource > resource_budget:
            self._log_with_iter("delta_resource = {}".format(delta_resource))

            while True:
                target_resource = resource - delta_resource
                tmp = self._select_best_pruning_point(target_resource)
                if tmp is None and delta_resource_decay >= 1:
                    self.logger.info("pruning process ends")
                    return
                delta_resource *= delta_resource_decay
                if tmp is not None:
                    resource = tmp
                    break

            self._log_with_iter("pruning progress: resource: {}; budget: {}".format(
                resource, resource_budget))
            self._log_with_iter("start short term finetune")
            self._short_term_finetune()
            self._log_with_iter("finish short term finetune")

            self.clean_tmp_files()

            self.iteration += 1

        self.logger.info("start long term finetune")
        self._long_term_finetune()
        self.logger.info("finish long term finetune")
        self.iteration = None

    def _get_ckpt_name(self, iteration: int):
        return "pruning_iter_{}".format(iteration)

    def _long_term_finetune(self):
        torch.cuda.empty_cache()
        self.finetune(self.model, "long")
        self._save_model(self.model, "pruning_result")

    def _short_term_finetune(self):
        torch.cuda.empty_cache()
        self.finetune(self.model, "short")
        self._save_model(self.model, self._get_ckpt_name(self.iteration))

    def _load_model(self, ckpt_name: str) -> dict:
        if hasattr(self, "model"):
            del self.model

        model = self.construct_model()
        self.model = PrunedNetwork(model, self.pruning_config)
        ckpt_path_noext = osp.join(self.work_dir, ckpt_name)
        self.model.load_pruning_state("{}.json".format(ckpt_path_noext))
        self.model.network.load_state_dict(
            torch.load("{}.pth".format(ckpt_path_noext))["state_dict"]
        )
        self.model.to("cpu")
        with open(osp.join(ckpt_path_noext + ".pkl"), "rb") as f:
            return pickle.load(f)

    def _save_model_nosync(
        self, model: PrunedNetwork, ckpt_name: str,
        resource: float, accuracy: float
    ):
        ckpt_path_noext = osp.join(self.work_dir, ckpt_name)
        torch.save(
            {"state_dict": model.state_dict()},
            "{}.pth".format(ckpt_path_noext)
        )
        model.save_pruning_state("{}.json".format(ckpt_path_noext))
        with open(osp.join(ckpt_path_noext + ".pkl"), "wb") as f:
            pickle.dump({
                "resource": resource,
                "accuracy": accuracy,
            }, f)

    def _save_model(self, model: PrunedNetwork, ckpt_name: str):
        resource = self.evaluate_resource(model)
        accuracy = self.evaluate_accuracy(model)
        if self.rank == 0:
            self._save_model_nosync(model, ckpt_name, resource, accuracy)
        dist.barrier()

    def _evaluate_pruning_point(self, pruning_point: str, target_resource: float) -> dict:
        """evaluate pruning point
        Args:
            pruning_point: pruning point to evaluate
            target_resource: target resource
        Returns:
            channels: #pruned channels
            accuracy: accuracy
            resource: final resource consumption (which should <= target_resource)
        """
        self._log_with_iter(
            "evaluating pruning point: {}".format(pruning_point))
        step = self.pruning_plan["step"]
        cout = self.model.get_op_config(pruning_point).cout
        min_channels_to_prune = cout - cout // step * step
        if min_channels_to_prune == 0:
            min_channels_to_prune = step

        new_model = deepcopy(self.model)
        new_model.to("cpu")
        new_model.prune(
            pruning_point,
            new_model.select_channel_idxs(pruning_point, min_channels_to_prune)
        )

        resource = None
        for i in range(min_channels_to_prune, cout, step):
            resource = self.evaluate_resource(new_model)
            if resource <= target_resource:
                break
            if i + step < cout:
                new_model.to("cpu")
                new_model.prune(
                    pruning_point,
                    new_model.select_channel_idxs(pruning_point, step)
                )

        if resource is None or resource > target_resource:
            return {key: None for key in ["channels", "accuracy", "resource"]}

        accuracy = self.evaluate_accuracy(new_model)
        ret = {"channels": i, "accuracy": accuracy, "resource": resource}
        self._log_with_iter(
            "pruning point {} evaluation result: {}".format(pruning_point, ret))
        return ret

    def _select_best_pruning_point(self, target_resource: float) -> Optional[float]:
        self._log_with_iter("select_best_pruning_point(target_resource={})".format(
            target_resource))

        candidate_info_dic_path = osp.join(
            self.work_dir, self._get_ckpt_name(self.iteration) + "_candidates.pkl")
        if osp.exists(candidate_info_dic_path):
            with open(candidate_info_dic_path, "rb") as f:
                candidate_info_dic = pickle.load(f)
        else:
            candidate_info_dic = {}

        for pruning_point in self.pruning_points:
            if pruning_point in candidate_info_dic:
                candidate = candidate_info_dic[pruning_point]
            else:
                candidate = self._evaluate_pruning_point(
                    pruning_point, target_resource)
            if candidate["channels"] is None:
                self._log_with_iter(
                    "skip pruning point {}".format(pruning_point))
            candidate_info_dic[pruning_point] = candidate
            if self.rank == 0:
                fd, path = mkstemp(dir=self.work_dir)
                with open(path, "wb") as f:
                    pickle.dump(candidate_info_dic, f)
                os.close(fd)
                # ensure atomic move
                os.rename(path, candidate_info_dic_path)

        best_pruning_point = None
        for pruning_point, info_dic in candidate_info_dic.items():
            if info_dic["accuracy"] is None:
                continue
            if best_pruning_point is None:
                best_pruning_point = pruning_point
            elif info_dic["accuracy"] > candidate_info_dic[best_pruning_point]["accuracy"]:
                best_pruning_point = pruning_point

        if best_pruning_point is None:
            self._log_with_iter("no pruning point matches the resource budget")
            return None

        self._log_with_iter(
            "select pruning point {}".format(best_pruning_point))

        best_candidate_info_dic = candidate_info_dic[best_pruning_point]
        channels, resource, accuracy = \
            best_candidate_info_dic["channels"], \
            best_candidate_info_dic["resource"], \
            best_candidate_info_dic["accuracy"]

        # enforce sync model through disk
        if self.rank == 0:
            self.model.to("cpu")
            self.model.prune(
                best_pruning_point,
                self.model.select_channel_idxs(
                    best_pruning_point, channels
                ))
            self._save_model_nosync(self.model, "tmp", resource, accuracy)
        dist.barrier()
        resource = self._load_model("tmp")["resource"]
        dist.barrier()
        if self.rank == 0:
            ckpt_path_noext = osp.join(self.work_dir, "tmp")
            for ext in ["pth", "json", "pkl"]:
                os.remove(ckpt_path_noext + "." + ext)

        return resource
