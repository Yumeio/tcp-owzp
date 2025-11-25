import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super(InputEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Transformer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        embed_dim: int, 
        num_heads: int, 
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        
        # Scaling factor theo kiến trúc gốc "Attention is All You Need"
        self.sqrt_embed_dim = math.sqrt(embed_dim)
        
        # 1. Embedding Layers
        # Encoder nhận 14 features
        self.encoder_embedding = InputEmbedding(input_dim, embed_dim, dropout)
        # Decoder nhận 4 features 
        self.decoder_embedding = InputEmbedding(output_dim, embed_dim, dropout)
        
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        # 2. Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True # Input dạng (Batch, Seq, Feature)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True # Input dạng (Batch, Seq, Feature)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 4. Output Layer
        self.fc_out = nn.Linear(embed_dim, output_dim)
        
        # Khởi tạo trọng số
        self.init_weights()

    def init_weights(self):
        """Khởi tạo Xavier Uniform cho các lớp Linear để hội tụ tốt hơn"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate_square_subsequent_mask(self, sz):
        """Tạo mặt nạ (Mask) để Decoder không nhìn thấy tương lai"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        """
        src: (Batch_Size, Src_Seq_Len, Input_Dim) -> Dữ liệu quá khứ (5 bước)
        tgt: (Batch_Size, Tgt_Seq_Len, Output_Dim) -> Dữ liệu để Decoder học (Shifted)
        """
        # --- 1. ENCODER ---
        # Embed + Scale + Positional Encoding
        src_emb = self.encoder_embedding(src) * self.sqrt_embed_dim
        src_emb = self.pos_encoder(src_emb)
        
        # Qua Encoder
        memory = self.encoder(src_emb)
        
        # --- 2. DECODER ---
        # Tạo mask cho Decoder (Causal Mask)
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # Embed Tgt
        tgt_emb = self.decoder_embedding(tgt) * self.sqrt_embed_dim
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Qua Decoder (cần Memory từ Encoder và Mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # --- 3. OUTPUT ---
        return self.fc_out(output)