"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import transformer_engine as te
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import (rmsnorm_fwd_fp8_inf,
                                                       rmsnorm_fwd_inf)
from vllm.model_executor.custom_op import CustomOp


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

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from vllm._ipex_ops import ipex_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO(woosuk): Implement an optimized kernel for GemmaRMSNorm.
        return self.forward_native(x, residual)


class RMSNormFp8Output(RMSNorm):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        output_scale: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(hidden_size=hidden_size, eps=eps)
        # Initialize FP8 meta tensor and other required objects
        fp8_meta_tensor = tex.FP8TensorMeta()
        self.fp8_meta_tensor = None
        if output_scale is not None:
            fp8_meta_tensor.scale = torch.tensor(output_scale,
                                                 dtype=torch.float32,
                                                 device="cuda")
            # Dummy tensor due to the underlying API requirements.
            fp8_meta_tensor.scale_inv = torch.tensor(output_scale,
                                                     dtype=torch.float32,
                                                     device="cuda")
            # Dummy tensor due to the underlying API requirements.
            fp8_meta_tensor.amax_history = torch.tensor([[0]],
                                                        dtype=torch.float32,
                                                        device="cuda")
            self.fp8_meta_tensor = fp8_meta_tensor

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x = x + residual
        sm_margin = 0
        if self.fp8_meta_tensor is not None:
            fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
            otype = tex.DType.kFloat8E4M3
            output = rmsnorm_fwd_fp8_inf(x, self.weight, self.variance_epsilon,
                                         self.fp8_meta_tensor, fp8_tensor,
                                         otype,
                                         sm_margin).view(torch.float8_e4m3fn)
        else:
            output = rmsnorm_fwd_inf(x, self.weight, self.variance_epsilon,
                                     sm_margin)
        if residual is not None:
            return output, residual
        return output
