"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.utils.cudnn_util import build_fuse_added_rmsnorm_fp8_graph, execute_graph, cudnn_reshape_to_3d

layernorm_graph_cache = {}


class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from vllm import _custom_ops as ops

        nvtx.range_push(str(x.shape))
        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            nvtx.range_pop()
            return x, residual

        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        nvtx.range_pop()
        return out

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class CudnnRMSNorm(RMSNorm):

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        graph_key = x.shape
        nvtx.range_push(str(graph_key))
        nvtx.range_push('var init')
        output = torch.empty((1, ) + x.shape, device=x.device, dtype=x.dtype)
        variance_epsilon = torch.tensor(self.variance_epsilon,
                                        device="cpu",
                                        dtype=float)
        output_scale = torch.empty([1], device="cuda", dtype=torch.float)
        nvtx.range_pop()
        nvtx.range_push('graph caching')
        if graph_key not in layernorm_graph_cache:
            results = build_fuse_added_rmsnorm_fp8_graph(
                x, self.weight, residual, variance_epsilon, output_scale)
            layernorm_graph_cache[graph_key] = results
        else:
            results = layernorm_graph_cache[graph_key]
        nvtx.range_pop()
        nvtx.range_push('variant pack')
        variant_pack = {
                results['act']:
                x.data_ptr(),
                results['weight']:
                self.weight.data_ptr(),
                results['residual']:
                residual.data_ptr() if residual is not None else None,
                results['variance_epsilon']:
                variance_epsilon,
                results['output_scale']:
                output_scale.data_ptr(),
                results['output']:
                output.data_ptr(),
        }
        nvtx.range_pop()
        nvtx.range_push('execute graph')
        execute_graph(results['graph'], variant_pack)
        nvtx.range_pop()
        nvtx.range_pop()
        if residual is not None:
            return output.reshape(x.shape), residual
        else:
            return output.reshape(x.shape)
