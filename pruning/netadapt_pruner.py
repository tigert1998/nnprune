import os.path as osp
import json
from copy import deepcopy
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from pruning.pruned_network import PrunedNetwork


class NetadaptPruner:
    """NetadaptPruner

    Args:
        unpruned_model: unpruned model
        model_constructor: A function that returns the skeleton of the model to prune
        pruning_config_path: pruning_config path for unpruned_model
        pruning_plan: A dict that defines the pruning process. Available keys:
            step: the alignment of channels for the pruned model;
            delta_resource: the least resource reduction in a round of pruning;
            resource_budget: resource budget;
            delta_resource_decay: delta_resource decay rate. Default: 1 (no decay)
        resource_evaluator: A function that returns the resource consumed by the model
        accuracy_evaluator: A function that returns the accuracy of the model
        finetuner: A function that finetunes the model. An additional string parameter
            will be passed to distinguish whether it is long term or short term finetune.
            The former one is "long" while the other is "short".
        work_dir: the work directory to save log and checkpoints
        tmp_file_cleaner: subroutine to clean tmp files after each iteration. Arguments:
            work_dir: work dir to clean;
            iteration: current iteration number.
        get_logger: A function that takes the following arguments and returns a logger:
            work_dir: work dir to save log;
            rank: the rank in the context of distributed training.
    """

    def __init__(
        self,
        unpruned_model: nn.Module,
        model_constructor: Callable[[], nn.Module],
        pruning_config_path: str,
        pruning_plan: dict,
        resource_evaluator: Callable[[PrunedNetwork], float],
        accuracy_evaluator: Callable[[PrunedNetwork], float],
        finetuner: Callable[[PrunedNetwork, str], None],
        work_dir: str,
        tmp_file_cleaner: Callable[[str, int], None],
        get_logger: Callable[[str, int], None],
    ):
        self.model_constructor = model_constructor
        self.pruning_config_path = pruning_config_path
        with open(self.pruning_config_path, "r") as f:
            self.pruning_config = json.load(f)

        self.model = PrunedNetwork(unpruned_model, self.pruning_config)
        self.pruning_points = list(self.model.dependencies.keys())

        self.pruning_plan = pruning_plan
        self.resource_evaluator = resource_evaluator
        self.accuracy_evaluator = accuracy_evaluator
        self.finetuner = finetuner
        self.work_dir = work_dir

        self.num_gpus = torch.cuda.device_count()
        self.rank, self.work_size = dist.get_rank(), dist.get_world_size()

        self.tmp_file_cleaner = tmp_file_cleaner
        self.logger = get_logger(self.work_dir, self.rank)
        self.iteration = None

    def run(self):
        resource = self.resource_evaluator(self.model)

        delta_resource = self.pruning_plan["delta_resource"]
        resource_budget = self.pruning_plan["resource_budget"]
        delta_resource_decay = self.pruning_plan.get("delta_resource_decay", 1)

        # resume progress
        self.iteration = 0
        while True:
            ckpt_name = self._get_ckpt_name(self.iteration)
            if not osp.exists(osp.join(self.work_dir, ckpt_name + ".pth")):
                if self.iteration >= 1:
                    self._load_model(self._get_ckpt_name(self.iteration - 1))
                break
            self.iteration += 1

        self.logger.info(
            "resume pruning from iteration #{}".format(self.iteration))

        while resource > resource_budget:
            self.logger.info("[iteration #{}]".format(self.iteration))

            target_resource = resource - delta_resource
            resource = self._select_best_pruning_point(target_resource)

            self.logger.info(
                "[iteration #{}] pruning progress: resource: {}; budget: {}".format(
                    self.iteration, resource, resource_budget
                ))
            self.logger.info(
                "[iteration #{}] start short term finetune".format(self.iteration))
            self._short_term_finetune()
            self.logger.info(
                "[iteration #{}] finish short term finetune".format(self.iteration))

            self.tmp_file_cleaner(self.work_dir, self.iteration)

            delta_resource *= delta_resource_decay
            self.iteration += 1

        self.logger.info("start long term finetune")
        self._long_term_finetune()
        self.logger.info("finish long term finetune")
        self.iteration = None

    def _get_ckpt_name(self, iteration: int):
        return "pruning_iter_{}".format(iteration)

    def _long_term_finetune(self):
        torch.cuda.empty_cache()
        self.finetuner(self.model, "long")
        self._save_model(self.model, "pruning_result")

    def _short_term_finetune(self):
        torch.cuda.empty_cache()
        self.finetuner(self.model, "short")
        self._save_model(self.model, self._get_ckpt_name(self.iteration))

    def _load_model(self, ckpt_name: str):
        if hasattr(self, "model"):
            del self.model

        model = self.model_constructor()
        self.model = PrunedNetwork(model, self.pruning_config)
        ckpt_path_noext = osp.join(self.work_dir, ckpt_name)
        self.model.load_pruning_state("{}.json".format(ckpt_path_noext))
        self.model.load_state_dict(
            torch.load("{}.pth".format(ckpt_path_noext))["state_dict"]
        )
        self.model.to("cpu")

    def _save_model(self, model: PrunedNetwork, ckpt_name: str):
        if self.rank == 0:
            ckpt_path_noext = osp.join(self.work_dir, ckpt_name)
            torch.save(
                {"state_dict": model.state_dict()},
                "{}.pth".format(ckpt_path_noext)
            )
            model.save_pruning_state("{}.json".format(ckpt_path_noext))
        dist.barrier()

    def _evaluate_pruning_point(self, pruning_point: str, target_resource: float) -> Tuple[PrunedNetwork, float, float]:
        """evaluate pruning point
        Args:
            pruning_point: pruning point to evaluate
        Returns:
            network: pruned network
            accuracy: accuracy
            resource: final resource consumption (which should <= target_resource)
        """
        step = self.pruning_plan["step"]
        cout = self.model.get_op_config(pruning_point).cout
        min_channels_to_prune = cout - cout // step * step
        if min_channels_to_prune == 0:
            min_channels_to_prune = step

        new_model = deepcopy(self.model)
        new_model.prune(
            pruning_point,
            new_model.select_channel_idxs(pruning_point, min_channels_to_prune)
        )

        for i in range(min_channels_to_prune, cout, step):
            resource = self.resource_evaluator(new_model)
            if resource <= target_resource:
                break
            if i + step < cout:
                new_model.prune(
                    pruning_point,
                    new_model.select_channel_idxs(pruning_point, step)
                )

        accuracy = self.accuracy_evaluator(new_model)
        return new_model, accuracy, resource

    def _select_best_pruning_point(self, target_resource: float) -> float:
        best_candidate: Optional[Tuple[int, PrunedNetwork, float, float]] \
            = None

        for idx, pruning_point in enumerate(self.pruning_points):
            model, accuracy, resource = \
                self._evaluate_pruning_point(pruning_point, target_resource)
            new_candidate = (idx, model, accuracy, resource)
            if best_candidate is None:
                best_candidate = new_candidate
            elif accuracy > best_candidate[2]:
                best_candidate = new_candidate

        idx = best_candidate[0]

        self.logger.info("[iteration #{}] select pruning point {}".format(
            self.iteration, self.pruning_points[idx]))

        self.model = best_candidate[1]
        return best_candidate[3]
