import torch
import torch.nn as nn

class iTransformer(nn.Module):
    def __init__(
        self,
        input_dim = 15,
        output_dim: int = 4,
        seq_len: int = 5,
        pred_len: int = 1,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layer: int = 4,
        dropout: float = 0.0
        ) -> None:

        super(iTransformer, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim

        # Embedding: Nhúng chuỗi thời gian (seq_len) thành vector (embed_dim)
        # Mỗi Feature sẽ có một vector đại diện riêng
        self.enc_embedding = nn.Linear(seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        ) # # batch_first=True -> Input: (Batch, features, embed_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer= encoder_layer,
            num_layers=num_layer
        )

        self.projector = nn.Linear(embed_dim, pred_len)

        self.init_weights()

    def init_weights(self):
        """Khởi tạo Xavier Uniform cho các lớp Linear để hội tụ tốt hơn"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, dec_in=None):
        """
        x shape: [Batch, Seq_Len, Num_Features] -> [? , 5, 15]
        """

        x = x.permute(0, 2, 1) # -> [?, 15, 5]

        enc_out = self.enc_embedding(x) # [?, 15, embed_dim]

        # Tìm mối liên hệ
        enc_out = self.encoder(enc_out) # Shape: [?, 15, embed_dim]

        dec_out = self.projector(enc_out) # Shape: [?, 15, 1]

        dec_out = dec_out.permute(0, 2, 1) # Shape: [?, 1, 15]
        output = dec_out[:, :, :self.output_dim] # [?, 1, 4]
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
