# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple

import numpy as np
import time
import torch
from omegaconf import OmegaConf

from mtrl.agent import grad_manipulation as grad_manipulation_agent
from mtrl.utils.types import ConfigType, TensorType
from scipy.optimize import least_squares
#from mtrl.agent.mgda import MinNormSolver


def _check_param_device(param: TensorType, old_param_device: Optional[int]) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError("Found two parameters on different devices, "
                            "this is currently not supported.")
    return old_param_device


def apply_vector_grad_to_parameters(vec: TensorType, parameters: Iterable[TensorType], accumulate: bool = False):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError("expected torch.Tensor, but got: {}".format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        if accumulate:
            param.grad = (param.grad + vec[pointer:pointer + num_param].view_as(param).data)
        else:
            param.grad = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


class Agent(grad_manipulation_agent.Agent):

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        agent_cfg: ConfigType,
        multitask_cfg: ConfigType,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """Regularized gradient algorithm."""
        agent_cfg_copy = deepcopy(agent_cfg)
        del agent_cfg_copy['fairgrad_alpha']

        OmegaConf.set_struct(agent_cfg_copy, False)
        agent_cfg_copy.cfg_to_load_model = None
        agent_cfg_copy.should_complete_init = False
        agent_cfg_copy.loss_reduction = "none"
        OmegaConf.set_struct(agent_cfg_copy, True)

        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            agent_cfg=agent_cfg_copy,
            device=device,
        )
        self.agent._compute_gradient = self._compute_gradient
        self._rng = np.random.default_rng()

        self.fairgrad_alpha = agent_cfg['fairgrad_alpha']
        self.wi_map = {}
        self.num_param_block = -1
        self.conflicts = []
        self.last_w = None
        self.save_target = 500000

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        #t0 = time.time()
        task_loss = self._convert_loss_into_task_loss(loss=loss, env_metadata=env_metadata)
        num_tasks = task_loss.shape[0]
        grad = []

        for index in range(num_tasks):
            grad.append(
                tuple(_grad.contiguous() for _grad in torch.autograd.grad(
                    task_loss[index],
                    parameters,
                    retain_graph=(retain_graph or index != num_tasks - 1),
                    allow_unused=allow_unused,
                )))

        grad_vec = torch.cat(
            list(map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)),
            dim=0,
        )  # num_tasks x dim

        regularized_grad = self.fairgrad(grad_vec, num_tasks)
        apply_vector_grad_to_parameters(regularized_grad, parameters)

    # def fairgrad(self, grads, num_tasks):
    #     """FairGrad
    #     Args:
    #         grads: tensor [num_tasks, dim]
    #     Return:
    #         g: manipulated gradient
    #     """
    #     alpha = self.fairgrad_alpha
    #     x_start = np.ones(num_tasks) / num_tasks
    #     GG = torch.mm(grads, grads.t()).cpu()
    #     A = GG.data.cpu().numpy()

    #     def objfn(x):
    #         # return np.power(np.dot(A, x), alpha) - 1 / x
    #         return np.dot(A, x) - np.power(1 / x, 1 / alpha)

    #     res = least_squares(objfn, x_start, bounds=(0, np.inf))
    #     w_cpu = res.x
    #     ww = torch.Tensor(w_cpu).to(grads.device)
    #     g = (grads * ww.view(-1, 1)).sum(0)
    #     return g
    
    def fairgrad(self, grads, num_tasks):
        """FairGrad with SGD
        Args:
            grads: tensor [num_tasks, dim]
        Return:
            g: manipulated gradient
        """
        alpha = self.fairgrad_alpha
        w = torch.ones(num_tasks) / num_tasks
        GG = torch.mm(grads, grads.t()).cpu()
        
        niter = 20
        w_best = None
        obj_best = np.inf
        w.requires_grad = True
        optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.5)
        for i in range(niter):
            optimizer.zero_grad()
            obj = torch.linalg.norm(
                torch.mv(GG, w) - torch.pow(1 / (w + 1e-8), 1 / alpha)
            ) ** 2
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(w, max_norm=1.0)
            optimizer.step()
            w.data.clamp_(min=1e-8)
        w.requires_grad = False
        ww = w_best.to(grads.device)
        g = (grads * ww.view(-1, 1)).sum(0)
        return g
