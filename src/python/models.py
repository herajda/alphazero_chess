import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from chess_game import ChessGame

class ChessModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.board_size = ChessGame.N
        self.num_actions = ChessGame.ACTIONS
        self.initial_channels = 119

    def forward(self, x):
        raise NotImplementedError

class TransformerModel(ChessModel):
    def __init__(self, args):
        super().__init__(args)
        self.dim_model = getattr(args, 'dim_model', 512)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.num_heads = getattr(args, 'num_heads', 8)
        self.ff_multiplier = 2

        # --- Input projection ---
        self.input_proj = nn.Conv2d(
            self.initial_channels, self.dim_model, kernel_size=1
        )

        # --- 2D Positional Encoding ---
        self.register_buffer("pos_encoding", self.create_positional_encoding())

        # --- Transformer stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_model * self.ff_multiplier,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        self.use_policy_1x1 = True
        # --- Policy head ---
        self.policy_1x1 = nn.Conv2d(self.dim_model, 73, kernel_size=1, bias=False)
        
        # --- Value head ---
        self.value_conv = nn.Conv2d(self.dim_model, 1, kernel_size=3, padding=1)
        self.value_flatten = nn.Flatten()
        self.value_dense = nn.Linear(self.board_size * self.board_size, 1)

    def create_positional_encoding(self):
        pe = torch.zeros(self.board_size, self.board_size, self.dim_model)
        pos_row = torch.arange(self.board_size).float().unsqueeze(1)
        pos_col = torch.arange(self.board_size).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2).float() *
            (-math.log(10000.0) / self.dim_model)
        )

        pe_row = torch.zeros(self.board_size, self.dim_model)
        pe_col = torch.zeros(self.board_size, self.dim_model)
        pe_row[:, 0::2] = torch.sin(pos_row * div_term)
        pe_row[:, 1::2] = torch.cos(pos_row * div_term)
        pe_col[:, 0::2] = torch.sin(pos_col * div_term)
        pe_col[:, 1::2] = torch.cos(pos_col * div_term)

        for i in range(self.board_size):
            for j in range(self.board_size):
                pe[i, j] = pe_row[i] + pe_col[j]

        pe = pe.view(-1, self.dim_model).unsqueeze(0)
        return pe

    def forward(self, x):
        bsz = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = self.input_proj(x)

        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_encoding
        x = self.transformer(x)

        x = x.transpose(1, 2).view(bsz, self.dim_model, self.board_size, self.board_size)

        # Policy Head
        logits_73_8x8 = self.policy_1x1(x)
        policy_logits = logits_73_8x8.permute(0, 2, 3, 1).contiguous()
        policy_logits = policy_logits.view(bsz, 64*73)

        # Value Head
        vx = self.value_conv(x)
        vx = self.value_flatten(vx)
        value = torch.tanh(self.value_dense(vx))

        return policy_logits, value

class CNNModel(ChessModel):
    def __init__(self, args):
        super().__init__(args)
        self.num_filters = getattr(args, 'num_filters', 256)
        self.num_layers = getattr(args, 'num_layers', 10) # Number of conv blocks

        self.conv_input = nn.Sequential(
            nn.Conv2d(self.initial_channels, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU()
        )

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU()
            ) for _ in range(self.num_layers)
        ])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 73, kernel_size=1),
            nn.BatchNorm2d(73),
            nn.ReLU(),
            nn.Flatten(),
            # No linear layer here, we map directly to 8x8x73
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        bsz = x.size(0)
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        x = self.conv_input(x)

        for layer in self.conv_layers:
            x = layer(x)

        # Policy
        policy_logits = self.policy_head(x) # [B, 73*64]
        
        # Value
        value = self.value_head(x)

        return policy_logits, value

class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ResNetModel(ChessModel):
    def __init__(self, args):
        super().__init__(args)
        self.num_filters = getattr(args, 'num_filters', 256)
        self.num_blocks = getattr(args, 'num_layers', 10) # Reusing num_layers arg for blocks

        self.conv_input = nn.Sequential(
            nn.Conv2d(self.initial_channels, self.num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList([
            ResBlock(self.num_filters) for _ in range(self.num_blocks)
        ])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 73, kernel_size=1), # AlphaZero uses 2 filters here usually, but we stick to 73 for direct map
            nn.BatchNorm2d(73),
            nn.ReLU(),
            nn.Flatten()
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_input(x)

        for block in self.res_blocks:
            x = block(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value
