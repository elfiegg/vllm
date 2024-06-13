from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import cudnn
import torch.cuda.nvtx as nvtx
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.utils.cudnn_util import build_fp8_matmul_bias_graph, execute_graph, cudnn_reshape_to_3d, build_fp8_matmul_bias_silu_graph
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import print_warning_once

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


def cutlass_fp8_supported() -> bool:
    capability = torch.cuda.get_device_capability()
    capability = capability[0] * 10 + capability[1]
    major, minor = torch.version.cuda.split(".")
    version = int(major) * 10 + int(minor)

    # CUTLASS FP8 kernels need at least
    #   CUDA 12.0 on SM90 systems (Hopper)
    #   CUDA 12.4 on SM89 systems (Lovelace)
    gpu_is_supported = False
    if capability >= 90:
        gpu_is_supported = version > 120
    elif capability >= 89:
        gpu_is_supported = version > 124

    return gpu_is_supported


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            return Fp8LinearMethod(self)
        if isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def _create_scale_param(
        self,
        scale_name: str,
        layer: torch.nn.Module,
        output_partition_sizes: List[int],
        **extra_weight_attrs,
    ) -> None:
        scale = Parameter(torch.empty(len(output_partition_sizes),
                                      dtype=torch.float32),
                          requires_grad=False)
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(
            scale, {
                **extra_weight_attrs,
                "fp8_scales_shard_indexer":
                self.scales_shard_indexer,
            })

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.process_after_load = True
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=weight_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        })

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            self._create_scale_param(
                scale_name="weight_scale",
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                **extra_weight_attrs)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                self._create_scale_param(
                    scale_name="input_scale",
                    layer=layer,
                    output_partition_sizes=output_partition_sizes,
                    **extra_weight_attrs)

    def scales_shard_indexer(
            self, param: torch.Tensor, loaded_weight: torch.Tensor,
            shard_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv_idxs = {"q": 0, "k": 1, "v": 2}

        if isinstance(shard_id, int):
            pass
        elif isinstance(shard_id, str):
            if shard_id not in qkv_idxs:
                raise ValueError(f"Unknown shard_id: {shard_id}")
            shard_id = qkv_idxs[shard_id]
        else:
            ValueError(f"Shard id must be int or str but got {type(shard_id)}")

        return param[shard_id], loaded_weight

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        # If checkpoint is fp/bf16 (not serialized fp8), quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.logical_widths = None
            layer.input_scale = None
            return

        # If checkpoint is fp8, requantize the separately quantized logical
        # weights into a single fp8 weight with a single weight scale.
        else:
            # WEIGHT_SCALE / WEIGHT
            #   Loop over logical weights, requantizing with single scale.
            max_w_scale = layer.weight_scale.max()
            start = 0
            for idx, logical_width in enumerate(layer.logical_widths):
                end = start + logical_width
                weight_dq = per_tensor_dequantize(layer.weight[start:end, :],
                                                  layer.weight_scale[idx])

                layer.weight[start:end, :] = per_tensor_quantize(
                    weight_dq, layer.weight_scale.max())
                start = end
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

            # WEIGHT
            #   Transpose weight for passing to torch._scaled_mm
            weight = layer.weight
            layer.weight = Parameter(weight.t(), requires_grad=False)

            # INPUT ACTIVATION SCALE
            #   Dynamic: set to None (required input to ops.scaled_fp8_quant).
            #   Static:  set to max of the input_scales (since they are equal).
            if self.quant_config.activation_scheme == "dynamic":
                layer.input_scale = None
            elif self.quant_config.activation_scheme == "static":
                if not all_close_1d(layer.input_scale):
                    raise ValueError(
                        "All the input_scales for the logical weights of a "
                        f"layer must be equal. But got {layer.input_scale}")
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)
            else:
                raise ValueError(
                    f"Unknown scheme {self.quant_config.activation_scheme}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              fuse_silu=False) -> torch.Tensor:
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.
        graph_key = (layer.weight.shape, x.shape)
        nvtx.range_push(str(graph_key))
        if bias is None and self.cutlass_fp8_supported:
            qinput, x_scale = ops.scaled_fp8_quant(x, layer.input_scale)

            # Fused GEMM_DQ
            output = ops.cutlass_scaled_mm_dq(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
            )

        else:
            qinput, x_scale = ops.scaled_fp8_quant(x,
                                                   layer.input_scale,
                                                   batch_dim_padding=17)

            # Fused GEMM_DQ -- note we padded the input above because
            # torch._scaled_mm is more performant for matrices with
            # batch dimension > 16. Note that this could change
            # in the future.
            output, _ = torch._scaled_mm(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                bias=bias,
            )
        nvtx.range_pop()
        return torch.narrow(output, 0, 0, x.shape[0])


class Fp8KVCacheMethod(QuantizeMethodBase):
    """Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """Create "weight" (aka kv_scale) for an attention layer.

        Args:
            layer: The layer that is using the QuantizeMethodBase factory.
        """
        # Initialize the KV cache scale to 1.0 as the default value.
        # If the kv_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        layer.kv_scale = Parameter(torch.tensor(1.0), requires_grad=False)

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError("Fp8KVCacheMethod.apply should not be called.")

    def process_weights_after_loading(self, layer: Module) -> None:
        # If the kv-cache dtype is auto, we enforce the kv-scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        if layer.kv_cache_dtype != "auto":
            kv_scale = layer.kv_scale.to("cpu").tolist()
            if not isinstance(kv_scale, float):
                raise ValueError("Only support per-tensor scaling factor "
                                 "for fp8 KV cache")
            layer._kv_scale = kv_scale
            if layer._kv_scale == 1.0 and "e5m2" not in layer.kv_cache_dtype:
                print_warning_once(
                    "Using KV cache scaling factor 1.0 for fp8_e4m3. This may "
                    "cause accuracy issues. Please make sure kv-cache scaling "
                    "factor is available in the fp8 checkpoint.")
        del layer.kv_scale


class CudnnFp8Config(Fp8Config):

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.cudnn_graphs = {}
        self.cudnn_handle = cudnn.create_handle()
    
    @classmethod
    def get_name(cls) -> str:
        return "cudnn-fp8"
        
    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            return CudnnFp8LinearMethod(self)
        if isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


class CudnnFp8LinearMethod(Fp8LinearMethod):
    """Linear method for FP8 with cuDNN fusion support."""

    def __init__(self, quant_config: CudnnFp8Config):
        self.quant_config = quant_config

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              fuse_silu: bool = False,
              fp8_out_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dtype is not torch.float8_e4m3fn :
            # or self.quant_config.activation_scheme != 'static':
            qinput, x_scale = ops.scaled_fp8_quant(x, layer.input_scale)
        else:
            qinput = x
            x_scale = torch.empty(
                1, device=x.device, dtype=torch.float
            ) if layer.input_scale is None else layer.input_scale
        if not fuse_silu:
            # Fused GEMM_DQ
            output = ops.cutlass_scaled_mm_dq(
                qinput,
                layer.weight,
                out_dtype=torch.float16,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
            )
            return torch.narrow(output, 0, 0, x.shape[0])
        else:
            output_shape = (x.shape[0], layer.weight.shape[1] // 2)
            # dtype = torch.float8_e4m3fn if fp8_out_scale is not None else x.dtype
            dtype = torch.float8_e4m3fn
            output = torch.empty((1, ) + output_shape,
                                 device=x.device,
                                 dtype=dtype)
            d = layer.weight.shape[1] // 2
            weight_up = layer.weight[:, :d]
            weight_down = layer.weight[:, d:]

            nvtx.range_push('scalar multiplication')
            dq_scale = x_scale * layer.weight_scale
            nvtx.range_pop()

            graph_key = (layer.weight.shape, x.shape, fuse_silu)
            nvtx.range_push(str(graph_key))

            nvtx.range_push("fp8 out scale")
            fp8_out_scale = torch.empty(1,
                                        device=x.device,
                                        dtype=x_scale.dtype)
            nvtx.range_pop()

            nvtx.range_push('graph caching')
            if graph_key not in self.quant_config.cudnn_graphs:
                results = build_fp8_matmul_bias_silu_graph(
                    qinput, weight_up, weight_down, dq_scale, bias,
                    fp8_out_scale)
                self.quant_config.cudnn_graphs[graph_key] = results
            else:
                results = self.quant_config.cudnn_graphs[graph_key]
            nvtx.range_pop()

            nvtx.range_push('variant pack')
            variant_pack = {
                results['act']:
                qinput.data_ptr(),
                results['bias']:
                bias.data_ptr() if bias is not None else None,
                results['weight_up']:
                weight_up.data_ptr(),
                results['weight_down']:
                weight_down.data_ptr(),
                results['dq_scale']:
                dq_scale.data_ptr(),
                results['out_scale']:
                fp8_out_scale.data_ptr()
                if fp8_out_scale is not None else None,
                results['output']:
                output.data_ptr(),
            }
            nvtx.range_pop()

            nvtx.range_push('execute graph')
            execute_graph(results['graph'], variant_pack)
            nvtx.range_pop()
            nvtx.range_pop()
            return output.reshape(output_shape).narrow(0, 0, x.shape[0])


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def per_tensor_quantize(tensor: torch.Tensor,
                        inv_scale: Union[float, torch.Tensor]) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight
