import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.model_patcher import ModelPatcher

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quant_dtype=torch.int8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_dtype = quant_dtype
        
        # 存储量化后的权重（int8）和缩放因子（float32）
        self.register_buffer('quantized_weight', 
            torch.zeros(out_features, in_features, dtype=quant_dtype))
        self.register_buffer('weight_scale', 
            torch.ones(out_features, 1, dtype=torch.float32))
        
        # 可选偏置（保持原精度）
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.bias = None

    def quantize_weight(self, weight: torch.Tensor):
        """将float权重量化为int8"""
        if self.quant_dtype == torch.int8:
            # 对称量化：scale = max(abs(weight)) / 127
            scale = weight.abs().max(dim=1, keepdim=True)[0] / 127.0
            scale = torch.clamp(scale, min=1e-12)
            quantized = torch.round(weight / scale).to(torch.int8)
        elif self.quant_dtype == torch.float8_e4m3fn:
            # 直接类型转换（FP8动态范围）
            quantized = weight.to(torch.float8_e4m3fn)
            scale = torch.ones_like(scale)  # FP8无需缩放
        else:
            raise NotImplementedError(f"不支持的量化类型: {self.quant_dtype}")
            
        return quantized, scale

    def forward(self, x: torch.Tensor):
        """前向计算：实时反量化权重"""
        # 反量化为计算精度（float16/float32）
        weight = self.quantized_weight.to(x.dtype)
        if self.quant_dtype == torch.int8:
            weight = weight * self.weight_scale
        
        return F.linear(x, weight, self.bias)

def replace_linear_recursive(module, target_class, quant_dtype, module_name_to_exclude=None):
    """递归替换所有nn.Linear层"""
    if module_name_to_exclude is None:
        module_name_to_exclude = ["final_layer"]  # 可选排除特定层
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name not in module_name_to_exclude:
            # 创建量化层并复制权重
            q_linear = target_class(child.in_features, child.out_features, 
                                   child.bias is not None, quant_dtype)
            
            # 量化权重
            quantized_weight, scale = q_linear.quantize_weight(child.weight.data)
            q_linear.quantized_weight.data = quantized_weight
            if quant_dtype == torch.int8:
                q_linear.weight_scale.data = scale
            
            # 复制偏置
            if child.bias is not None:
                q_linear.bias.data = child.bias.data
            
            setattr(module, name, q_linear)
        else:
            replace_linear_recursive(child, target_class, quant_dtype, module_name_to_exclude)

class WeightOnlyQuantizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "quant_dtype": (["int8", "float8_e4m3fn", "float8_e5m2"], 
                               {"default": "int8"}),
                "exclude_layers": ("STRING", {"default": "final_layer,proj_out"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("quantized_model",)
    FUNCTION = "apply_quantization"
    CATEGORY = "Model Quantization/Advanced"
    
    def apply_quantization(self, model, quant_dtype, exclude_layers):
        if quant_dtype == "int8":
            dtype = torch.int8
        else:
            dtype = getattr(torch, quant_dtype)
        
        # 获取模型对象
        model_patcher = model.model
        exclude_list = [x.strip() for x in exclude_layers.split(",") if x.strip()]
        
        # 应用量化替换
        replace_linear_recursive(model_patcher.diffusion_model, 
                                QuantizedLinear, dtype, exclude_list)
        
        print(f"[WeightOnlyQuantize] 完成量化，格式: {quant_dtype}")
        return (model,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "WeightOnlyQuantize": WeightOnlyQuantizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeightOnlyQuantize": "Weight-Only Quantize (Linear)",
}