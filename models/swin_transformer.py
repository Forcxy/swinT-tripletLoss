import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from swin_triple.utils.config import config
from swin_triple.models.model import swin_tiny_patch4_window7_224
import torch
import os


class SwinFeatureExtractor(nn.Module):
    def __init__(self, model_name="swin_tiny", embedding_size=512, pretrained=True):
        super().__init__()
        # Backbone: Swin-Tiny (保持原结构)
        self.backbone = swin_tiny_patch4_window7_224(num_classes=0)  # 移除分类头

        # 投影头 (将特征映射到嵌入空间)、不加relu
        in_features = self.backbone.num_features
        self.projection = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        """
        输入:
            x: [B, 3, 224, 224]
        输出:
            normalized_embedding: [B, embed_dim] (L2归一化后的特征)
        """
        # 1. Patch Embedding
        x, H, W = self.patch_embed(x)  # [B, 56 * 56, embed_dim]
        x = self.pos_drop(x)

        # 2. Swin Layers
        for layer in self.layers:
            x, H, W = layer(x, H, W)  # 逐步下采样 [B, 56 * 56, 96] -> [B, 28 * 28, 192] -> ... -> [B, 7 * 7, 768]

        # 3. 全局特征提取
        x = self.norm(x)  # [B, 49, 768]
        x = x.mean(dim=1)  # 全局平均池化 [B, 768]

        # 4. 投影到嵌入空间
        return F.normalize(self.projection(x), p=2, dim=1)  # [B, embed_dim]




class SwinFeatureExtractor1(nn.Module):
    def __init__(self, model_name="swin_tiny", pretrained=True, embedding_size=512):
        super().__init__()

        # 禁用自动下载
        self.backbone = swin_tiny_patch4_window7_224(pretrained=False)

        if pretrained and config.USE_LOCAL_WEIGHTS:
            print(f"Loading local weights from {config.PRETRAINED_WEIGHTS}")

            # 检查权重文件是否存在
            if not os.path.exists(config.PRETRAINED_WEIGHTS):
                raise FileNotFoundError(
                    f"Pretrained weights not found at {config.PRETRAINED_WEIGHTS}\n"
                    "Please download from: "
                    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
                )

            # 加载本地权重
            state_dict = torch.load(config.PRETRAINED_WEIGHTS, map_location="cpu")

            # 处理不同的权重文件格式
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # 过滤不需要的键（例如分类头）
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}

            # 获取当前模型的状态字典
            model_dict = self.backbone.state_dict()

            # 初始化匹配和不匹配的层记录
            matched_layers = []
            mismatched_layers = []

            # 筛选匹配的权重
            matched_state_dict = {}
            for name, param in state_dict.items():
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        matched_state_dict[name] = param
                        matched_layers.append(name)
                    else:
                        mismatched_layers.append((name, "shape mismatch",
                                                  f"pretrained: {param.size()}, model: {model_dict[name].size()}"))
                else:
                    mismatched_layers.append((name, "not in model", None))

            # 打印详细的匹配信息
            print("\n" + "=" * 50)
            print("权重加载报告:")
            print("-" * 50)
            print(f"成功匹配 {len(matched_layers)} 个层:")
            for layer in matched_layers:
                print(f"  ✓ {layer} (shape: {state_dict[layer].size()})")

            print("\n" + "-" * 50)
            print(f"跳过 {len(mismatched_layers)} 个不匹配的层:")
            for layer, reason, info in mismatched_layers:
                if reason == "shape mismatch":
                    print(f"  ✗ {layer}: {info}")
                else:
                    print(f"  ✗ {layer}: 该层在模型中不存在")

            # 更新模型权重
            model_dict.update(matched_state_dict)
            load_result = self.backbone.load_state_dict(model_dict, strict=False)

            # 打印PyTorch的加载结果
            print("\n" + "-" * 50)
            print("PyTorch加载结果:")
            print(f"缺失的层: {load_result.missing_keys}")
            print(f"意外的层: {load_result.unexpected_keys}")
            print("=" * 50 + "\n")


        # 移除分类头
        if hasattr(self.backbone, "head"):
            del self.backbone.head

        # 添加投影头 不使用relu，保持特征的连续性
        in_features = self.backbone.num_features
        self.projection = nn.Linear(in_features, embedding_size)
        
    def forward(self, x):
        # 获取特征
        x = self.backbone.forward_features(x)
        
        # 全局平均池化
        if x.dim() == 3:  # [B, L, C]
            x = x.mean(dim=1)
        
        # 投影到嵌入空间
        x = self.projection(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        
        return x