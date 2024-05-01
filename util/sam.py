# Adapted from David Samuel's excellent implementation of Sharpness Aware Minimization (SAM) in PyTorch
# https://github.com/davda54/sam/tree/main
# As well as MosaicML's SAM implementation 
# https://github.com/mosaicml/composer/blob/dev/composer/algorithms/sam/sam.py
# and MosaicML's custom GradScaler supporting mixed precision training for closures to enable SAM with AMP
# https://github.com/mosaicml/composer/blob/dev/composer/trainer/_scaler.py 

from collections import defaultdict
from typing import Optional, Union

import torch
from packaging import version
from torch.cuda.amp.grad_scaler import GradScaler, OptState
import torch.distributed
from torch.optim import Optimizer

if version.parse(torch.__version__) >= version.parse('2.3.0'):
    from torch.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore
else:
    from torch.cuda.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore
from torch.nn.modules.batchnorm import _BatchNorm


# BatchNorm helper functions recommended by davda54
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


###################################
# davda54's implementation of SAM #
###################################
class SAM(torch.optim.Optimizer):
    def __init__(
            self, 
            base_optimizer, 
            rho=0.05, 
            adaptive=False, 
            **kwargs
        ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        defaults = {'rho': rho, 'adaptive': adaptive, **kwargs}
        super(SAM, self).__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: 
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


##################################
# MosaicML implementation of SAM #
##################################
class SAMOptimizer(torch.optim.Optimizer):
    """
    MosaicML implementation of Sharpness Aware Minimization 
    by Foret et al, 2020 https://arxiv.org/abs/2010.01412
    Implementation based on https://github.com/davda54/sam

    Args:
        base_optimizer (torch.optim.Optimizer) The optimizer to apply SAM to.
        rho (float, optional): The SAM neighborhood size. Must be greater than 0. Default: ``0.05``.
        epsilon (float, optional): A small value added to the gradient norm for numerical stability. Default: ``1.0e-12``.
        interval (int, optional): SAM will run once per ``interval`` steps. A value of 1 will
            cause SAM to run every step. Steps on which SAM runs take
            roughly twice as much time to complete. Default: ``1``.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        epsilon: float = 1.0e-12,
        interval: int = 1,
        **kwargs,
    ):
        if rho < 0:
            raise ValueError(f'Invalid rho, should be non-negative: {rho}')
        self.base_optimizer = base_optimizer
        self.global_step = 0
        self.interval = interval
        self._step_supports_amp_closure = True  # Flag for Composer trainer
        defaults = {'rho': rho, 'epsilon': epsilon, **kwargs}
        super(SAMOptimizer, self).__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def sub_e_w(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' not in self.state[p]:
                    continue
                e_w = self.state[p]['e_w']  # retrieve stale e(w)
                p.sub_(e_w)  # get back to "w" from "w + e(w)"

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + group['epsilon'])
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                p.sub_(self.state[p]['e_w'])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def step(self, closure=None):
        assert closure is not None, 'Sharpness Aware Minimization requires closure, but it was not provided'
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        loss = None

        if (self.global_step + 1) % self.interval == 0:
            # Compute gradient at (w) per-GPU, and do not sync
            loss = closure(ddp_sync=False)  # type: ignore
            if loss:
                self.first_step()  # Compute e(w) and set weights to (w + (e(w)) separately per-GPU
                loss_dict = {}  # Dummy loss dict to ignore loss logging from w + e(w)
                if closure(loss_dict=loss_dict):  # type: ignore Compute gradient at (w + e(w))
                    self.second_step()  # Reset weights to (w) and step base optimizer
                else:
                    self.sub_e_w()  # If second forward-backward closure fails, reset weights to (w)
        else:
            loss = closure()
            if loss:
                self.base_optimizer.step()

        self.global_step += 1
        return loss

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None
            ]),
            p='fro',
        )
        return norm
    
############################################################
# MosaicML's custom AMP-compatible GradScaler for closures #
############################################################
class ClosureGradScaler(GradScaler):
    """ClosureGradScaler allows for gradient scaling during with closures.

    We use closures with optimizers (see `here <https://pytorch.org/docs/stable/optim.html>`__)
    during training in order to support certain algorithms like
    :class:`~composer.algorithms.SAM`. This class allows us to perform gradient
    scaling (see `here <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`__)
    along with the use of closures during training.

    Args:
        ddp_reduce_scalar_and (Callable[[bool], bool]): A function that performs a
            ddp reduction with an `and` operation. Used to determine whether
            or not to continue computing an optimizer's `step` based on the presence
            of `inf/nan` in the gradients.
        ddp_reduce_tensor_sum (Callable[[Tensor], Tensor]): A function that performs
            a ddp reduction across tensors with a `sum` operation. Used to aggregate
            `inf/nan` information stored in tensors across devices.
    """

    def _force_scaler_ready(self, optimizer: Optimizer):
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        optimizer_state['stage'] = OptState.READY

    def _empty_all_grads(self, optimizer):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = None

    def _unscale_grads_and_continue(self, optimizer: Optimizer):
        if (not self._enabled):
            return True
        self._check_scale_growth_tracker('step')
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state['stage'] is OptState.STEPPED:
            raise RuntimeError('step() has already been called since the last update().')

        if optimizer_state['stage'] is OptState.READY:
            self.unscale_(optimizer)
        inf_detected = sum(v.item() for v in optimizer_state['found_inf_per_device'].values())
        return not inf_detected

    def step(self, optimizer: Optimizer, *args, **kwargs):
        """Step the optimizer with amp.

        Always called before the optimizer step. Checks if the optimizer can handle AMP closures (currently only
        Composer's SAM optimizer) If so, it passes an AMP-modified closure to the optimizer.
        """
        closure = kwargs['closure']

        def _amp_closure(**kwargs):
            self._force_scaler_ready(optimizer)
            self._empty_all_grads(optimizer)

            retval: float = closure(**kwargs)

            should_continue = self._unscale_grads_and_continue(optimizer)
            other_should_continue = torch.distributed.all_gather_object(should_continue)

            return retval if all(other_should_continue) else None

        return optimizer.step(closure=_amp_closure)  # type: ignore

    # Mostly copied from original grad_scaler implementation
    # See: https://pytorch.org/docs/stable/_modules/torch/amp/grad_scaler.html#GradScaler
    def update(self, new_scale: Optional[Union[float, torch.FloatTensor]] = None):
        """Updates the scale factor.

        If any optimizer steps were skipped, the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` non-skipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly; it is used to fill GradScaler's internal scale tensor. So, if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale that the GradScaler uses internally.)

        .. warning::

            This method should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.

        Args:
            new_scale (float | FloatTensor, optional):  New scale factor. (default: ``None``)
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker('update')

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = 'new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False.'
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state['found_inf_per_device'].values()
            ]

            assert len(found_infs) > 0, 'No inf checks were recorded prior to update.'

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            # This is the only line changed from original grad_scaler implementation
            torch.distributed.all_reduce(found_inf_combined, reduce_operation='SUM')

            torch._amp_update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
            )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)