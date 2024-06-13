from typing import Optional, Tuple
import torch
import cudnn
import torch.cuda.nvtx as nvtx


def cudnn_reshape_to_3d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensures a tensor has exactly three dimensions by adding leading singleton dimensions if necessary.
    """
    if isinstance(tensor, float):
        # Convert float to a tensor with shape [1, 1, 1]
        input_data = torch.tensor([[[tensor]]])
        return input_data

    required_dims = 3 - tensor.dim()
    return tensor.reshape((1, ) * required_dims +
                          tensor.shape) if tensor.dim() < 3 else tensor


def _to_cudnn_type(dtype: torch.dtype) -> cudnn.data_type:
    """
    Maps PyTorch data types to cuDNN data types.
    """
    type_mapping = {
        torch.float16: cudnn.data_type.HALF,
        torch.float32: cudnn.data_type.FLOAT,
        torch.float64: cudnn.data_type.DOUBLE,
        torch.int8: cudnn.data_type.INT8,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
        torch.uint8: cudnn.data_type.UINT8,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
    }
    cudnn_type = type_mapping.get(dtype)
    if cudnn_type is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return cudnn_type


def _build_graph_common(tensor_details, graph):
    """
    General helper to add tensors to a cuDNN graph.
    """
    tensors = {}
    for name, tensor in tensor_details.items():
        tensor = cudnn_reshape_to_3d(tensor) if tensor is not None else None
        tensors[name] = graph.tensor(
            name=name,
            dim=tensor.size(),
            stride=tensor.stride(),
            data_type=_to_cudnn_type(
                tensor.dtype)) if tensor is not None else None
    return tensors


def _build_fp8_matmul_bias_graph(x: torch.Tensor, weight: torch.Tensor,
                                 x_scale: torch.Tensor,
                                 weight_scale: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 graph: cudnn.pygraph) -> Tuple:
    """
    Builds a computation graph for FP8 matrix multiplication with optional bias addition.
    """
    tensors = _build_graph_common(
        {
            'act': x,
            'weight': weight,
            'act_scale': x_scale,
            'weight_scale': weight_scale,
            'bias': bias,
        }, graph)

    out_intermediate = graph.matmul(name="matmul",
                                    A=tensors['act'],
                                    B=tensors['weight'])
    scaled_out_intermediate = graph.mul(name="mul_act_scale",
                                        a=out_intermediate,
                                        b=tensors['act_scale'])
    final_scaled_output = graph.mul(name="mul_weight_scale",
                                    a=scaled_out_intermediate,
                                    b=tensors['weight_scale'])

    final_output = graph.bias(
        name="bias_add", input=final_scaled_output,
        bias=tensors.get('bias')) if bias is not None else final_scaled_output
    final_output.set_name("out").set_output(True).set_data_type(
        cudnn.data_type.HALF)
    result_dict = {**tensors, "output": final_output}
    return result_dict


def _build_fp8_matmul_bias_graph(x: torch.Tensor, weight: torch.Tensor,
                                 x_scale: torch.Tensor,
                                 weight_scale: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 graph: cudnn.pygraph) -> Tuple:
    """
    Creates a cuDNN graph for FP8 matrix multiplication with bias.
    """
    tensors = _build_graph_common(
        {
            'act': x,
            'weight': weight,
            'act_scale': x_scale,
            'weight_scale': weight_scale,
            'bias': bias,
        }, graph)

    out_intermediate = graph.matmul(name="matmul",
                                    A=tensors['act'],
                                    B=tensors['weight'])
    scaled_out_intermediate = graph.mul(name="mul_act_scale",
                                        a=out_intermediate,
                                        b=tensors['act_scale'])
    final_scaled_output = graph.mul(name="mul_weight_scale",
                                    a=scaled_out_intermediate,
                                    b=tensors['weight_scale'])

    final_output = graph.bias(
        name="bias_add", input=final_scaled_output,
        bias=tensors.get('bias')) if bias is not None else final_scaled_output
    final_output.set_data_type(cudnn.data_type.HALF)
    final_output.set_name("out").set_output(True)
    result_dict = {**tensors, "output": final_output}
    return result_dict


def _build_fp8_matmul_bias_silu_graph(x: torch.Tensor, weight_up: torch.Tensor,
                                      weight_down: torch.Tensor,
                                      dq_scale: torch.Tensor,
                                      bias: Optional[torch.Tensor],
                                      out_scale: Optional[torch.Tensor],
                                      graph: cudnn.pygraph) -> Tuple:
    """
    Builds a computation graph for FP8 matrix multiplication with bias addition, 
    followed by SiLU activation, ouput fp8 results if needed.
    """
    tensors = _build_graph_common(
        {
            'act': x,
            'weight_up': weight_up,
            'weight_down': weight_down,
            'dq_scale': dq_scale,
            'bias': bias,
            'out_scale': out_scale,
        }, graph)
    tensors['weight_up'].set_stride(
        [weight_up.shape[1] * weight_up.shape[2], 1, weight_up.shape[1]])
    tensors['weight_down'].set_stride(
        [weight_down.shape[1] * weight_down.shape[2], 1, weight_down.shape[1]])
    out_up = graph.matmul(name="matmul",
                          A=tensors['act'],
                          B=tensors['weight_up'])
    out_up = out_up if bias is None else graph.bias(
        name="bias_up", input=out_up, bias=tensors['bias'])
    out_down = graph.matmul(name="matmul",
                            A=tensors['act'],
                            B=tensors['weight_down'])
    out_down = out_down if bias is None else graph.bias(
        name="bias_down", input=out_down, bias=tensors['bias'])
    scaled_out_up = graph.mul(name="scaled_matmul_up",
                              a=out_up,
                              b=tensors['dq_scale'])
    scaled_out_down = graph.mul(name="scaled_matmul_down",
                                a=out_down,
                                b=tensors['dq_scale'])
    silu_result = graph.swish(scaled_out_up, name="swish")
    silu_mul = graph.mul(name="silu_mul", a=silu_result, b=scaled_out_down)
    output_tensor = silu_mul if out_scale is None else graph.mul(
        name="dq_fp8", a=silu_mul, b=tensors['out_scale'])
    output_tensor.set_output(True).set_data_type(
        cudnn.data_type.
        FP8_E4M3 if out_scale is not None else _to_cudnn_type(dq_scale.dtype))

    result_dict = {**tensors, "output": output_tensor}
    return result_dict


def _build_graph_plans(graph: cudnn.pygraph) -> cudnn._compiled_module.pygraph:
    """
    Builds a cuDNN graph for execution or storage after validation and plan building.
    """
    try:
        graph.validate()
        graph.build_operation_graph()
    except RuntimeError:
        print(graph)

    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph


def build_fp8_matmul_bias_graph(x: torch.Tensor, weight: torch.Tensor,
                                x_scale: torch.Tensor,
                                weight_scale: torch.Tensor,
                                bias: Optional[torch.Tensor]) -> Tuple:
    """
    Builds a cuDNN graph for FP8 matrix multiplication with bias.
    """
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT,
                          compute_data_type=cudnn.data_type.FLOAT)
    tensors = _build_fp8_matmul_bias_graph(
        cudnn_reshape_to_3d(x), cudnn_reshape_to_3d(weight),
        cudnn_reshape_to_3d(x_scale), cudnn_reshape_to_3d(weight_scale),
        cudnn_reshape_to_3d(bias) if bias is not None else None, graph)

    result_dict = {**tensors, "graph": _build_graph_plans(graph)}
    return result_dict


def build_fp8_matmul_bias_silu_graph(
        x: torch.Tensor, weight_up: torch.Tensor, weight_down: torch.Tensor,
        dq_scale: torch.Tensor, bias: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor]) -> Tuple:
    """
    Builds the cudnn graph for FP8 matrix multiplication with bias and SiLU activation.
    """

    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT,
                          compute_data_type=cudnn.data_type.FLOAT)
    tensors = _build_fp8_matmul_bias_silu_graph(
        cudnn_reshape_to_3d(x), cudnn_reshape_to_3d(weight_up),
        cudnn_reshape_to_3d(weight_down), cudnn_reshape_to_3d(dq_scale),
        cudnn_reshape_to_3d(bias) if bias is not None else None,
        cudnn_reshape_to_3d(out_scale) if out_scale is not None else None,
        graph)
    result_dict = {**tensors, "graph": _build_graph_plans(graph)}
    return result_dict


def build_fuse_added_rmsnorm_fp8_graph(x: torch.Tensor, weight: torch.Tensor,
                                       residual: Optional[torch.Tensor],
                                       variance_epsilon: float,
                                       output_scale: Optional[torch.Tensor]):
    """
    Builds the fp8 rms norm graph.
    """
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT,
                          compute_data_type=cudnn.data_type.FLOAT)
    tensors = _build_graph_common(
        {
            'act': x,
            'weight': weight,
            'residual': residual,
            'variance_epsilon': variance_epsilon,
            'output_scale': output_scale
        }, graph)
    tensors['variance_epsilon'].set_is_pass_by_value(True)
    added_x = graph.add(name="add", a=tensors['act'],
                        b=tensors['residual']) if residual is not None else x
    if residual is not None:
        added_x.set_data_type(_to_cudnn_type(x.dtype))
    output_tensor, _ = graph.rmsnorm(
        name="RMS",
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
        input=tensors['act'],
        scale=tensors['weight'],
        bias=tensors['residual'],
        epsilon=tensors['variance_epsilon'],
    )
    output_tensor.set_data_type(cudnn.data_type.FLOAT).set_dim(
        tensors['act'].get_dim()).set_stride(tensors['act'].get_stride())
    output_tensor = graph.mul(
        name="mul_scale", a=output_tensor, b=tensors['output_scale']
    ) if output_scale is not None else output_tensor
    output_dtype = torch.float8_e4m3fn if output_scale is not None else _to_cudnn_type(
        x.dtype)
    output_tensor.set_output(True).set_data_type(output_dtype).set_dim(
        tensors['act'].get_dim()).set_stride(tensors['act'].get_stride())
    result_dict = {
        **tensors, "output": output_tensor,
        "graph": _build_graph_plans(graph)
    }
    return result_dict


def execute_graph(graph: bytes, variant_pack: dict) -> None:
    """
    Executes a pre-built cuDNN graph.
    """
    nvtx.range_push('workspace tensor')
    workspace = torch.empty(graph.get_workspace_size(),
                            device="cuda",
                            dtype=torch.uint8)
    nvtx.range_pop()
    graph.execute(variant_pack, workspace)
