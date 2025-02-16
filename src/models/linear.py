import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy


weight_storage = {} # forward weights for BP and FA
weight_b_storage = {} # backward weights for FA


class LinearFunction(torch.autograd.Function):
    """Implementation of backpropagation for linear layer"""
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        grad_weight = grad_output.t().matmul(input) if ctx.needs_input_grad[1] else None
        grad_bias = grad_output.sum(0).squeeze(0) if bias is not None and ctx.needs_input_grad[2] else None
        return grad_input, grad_weight, grad_bias


class LinearFAFunction(torch.autograd.Function):
    """Implementation of feedback alignment for linear layer"""
    @staticmethod
    def forward(ctx, input, weight, weight_b, bias=None):
        ctx.save_for_backward(input, weight, weight_b, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_b, bias = ctx.saved_tensors
        grad_input = grad_output.matmul(weight_b.to(grad_output.device)) if ctx.needs_input_grad[0] else None
        grad_weight = grad_output.t().matmul(input) if ctx.needs_input_grad[1] else None
        grad_bias = grad_output.sum(0).squeeze(0) if bias is not None and ctx.needs_input_grad[3] else None
        return grad_input, grad_weight, None, grad_bias

class Linear(nn.Module):
    """Linear layer with different backpropagation modes"""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 mode: str,
                 layer: int,
                 w_seed: int = -1,
                 b_seed: int = -1):
        """
        Initializes the Linear layer with specific configurations.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Flag to include a bias term.
            mode (str): Training mode, e.g., "BP", "FA".
            layer (int): Layer index used for weight storage.
            w_seed (int, optional): Seed for weight initialization via storage. 
                                    Default is -1, which initializes without using weight storage.
            b_seed (int, optional): Seed for feedback alignment weight initialization via storage. 
                                    Default is -1, which initializes without using weight storage.
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        self.weight = self.initialize_weight(self.in_features, self.out_features, layer, w_seed, type="forwards")
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        match self.mode:
            case "BP":
                self.weight_fa = None
            case "FA":
                self.weight_fa = self.initialize_weight(self.in_features, self.out_features, layer, b_seed, type="backwards")

    def initialize_weight(self, in_features, out_features, layer, seed, type):
        """Initialize weight with He initialization"""
        if seed == -1:
            if type == "forwards":
                return nn.Parameter(torch.Tensor(np.random.normal(0.0, np.sqrt(2 / in_features), (out_features, in_features))), requires_grad=True)
            elif type == "backwards":
                return nn.Parameter(torch.Tensor(np.random.normal(0.0, np.sqrt(2 / in_features), (out_features, in_features))), requires_grad=False)
            else:
                raise ValueError("Invalid type")
            
        else:
            if type == "forwards":
                if layer not in weight_storage:
                    weight_storage[layer] = {}
                if seed not in weight_storage[layer]:
                    initial_weight = np.random.normal(0.0, np.sqrt(2 / in_features), (out_features, in_features))
                    weight_storage[layer][seed] = copy.deepcopy(initial_weight)
                weight_param = nn.Parameter(torch.Tensor(weight_storage[layer][seed]), requires_grad=True)
                return weight_param
            elif type == "backwards":
                if layer not in weight_b_storage:
                    weight_b_storage[layer] = {}
                if seed not in weight_b_storage[layer]:
                    initial_weight = np.random.normal(0.0, np.sqrt(2 / in_features), (out_features, in_features))
                    weight_b_storage[layer][seed] = copy.deepcopy(initial_weight)
                weight_param = nn.Parameter(torch.Tensor(weight_b_storage[layer][seed]), requires_grad=False)
                return weight_param

    def forward(self, input):
        """Forward pass with different backpropagation modes"""
        match self.mode:
            case "BP":
                return LinearFunction.apply(input, self.weight, self.bias)
            case "FA":
                return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
