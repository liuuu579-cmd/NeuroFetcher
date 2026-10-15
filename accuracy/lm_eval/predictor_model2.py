# predictor_model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """标准的Transformer位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # <<< 核心修改: 强制使用 float32 进行计算以保证稳定 >>>
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PredictorTransformer(nn.Module):
    """
    一个Transformer模型，直接预测每个OPT层的注意力分数矩阵。

    输入: 原始token嵌入 [B, S, D_MODEL]
    输出: 预测的注意力分数 [B, NUM_LAYERS, S]
    """
    def __init__(self, input_dim: int, num_opt_layers: int, nhead: int = 8, num_encoder_layers: int = 4, d_hid: int = 2048, dropout: float = 0.1):
        """
        Args:
            input_dim (int): 输入嵌入的维度 (D_MODEL from OPT)
            num_opt_layers (int): 目标OPT模型的层数 (e.g., 40)
            nhead (int): 此预测模型中Transformer层的头数
            num_encoder_layers (int): 此预测模型中Transformer层的数量
            d_hid (int): 此预测模型中前馈网络的维度
            dropout (float): Dropout率
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # 输出头：将每个token的隐藏状态映射到40个层的分数
        self.output_head = nn.Linear(input_dim, num_opt_layers)
        
        self.input_dim = input_dim
        self.num_opt_layers = num_opt_layers

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): 输入嵌入，形状为 [B, S, D_MODEL]
            src_padding_mask (torch.Tensor, optional): 填充位置的掩码，
                                                      形状为 [B, S]。 True表示该位置是填充。
        
        Returns:
            torch.Tensor: 预测的注意力分数，形状为 [B, NUM_LAYERS, S]
        """
        # TransformerEncoder默认期望 [S, B, D], 但我们用了batch_first=True,所以输入是 [B, S, D]
        # Pytorch的TransformerEncoderLayer还没有完全支持位置编码在batch_first=True下的自动处理，我们手动调整一下
        src = src.permute(1, 0, 2) # -> [S, B, D]
        src = self.pos_encoder(src * math.sqrt(self.input_dim))
        src = src.permute(1, 0, 2) # -> [B, S, D]

        # 编码器处理
        # src_key_padding_mask需要 [B, S]
        output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask) 
        # output shape: [B, S, D_MODEL]
        
        # 通过线性头得到分数
        scores = self.output_head(output) # -> [B, S, NUM_LAYERS]
        
        # 交换维度以匹配目标 [B, NUM_LAYERS, S]
        scores = scores.permute(0, 2, 1) # -> [B, NUM_LAYERS, S]
        
        return scores