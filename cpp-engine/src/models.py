import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer (sinusoidal, no learnable params).
    
    Formula:
        PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1)   = cos(pos / 10000^(2i/d_model))
    
    where pos is the position in the sequence, i is the dimension index.
    """
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        
        # Register as buffer (not a param, moves with model but not updated)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        
        Returns:
            (B, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DeepLOB(nn.Module):
    """
    Deep Learning model for Limit Order Book (LOB) mid-price prediction.

    Architecture:
        1. CNN Block  — extract spatial features across the 40 LOB feature columns
                        while preserving the 100-step time dimension.
        2. LSTM Block — capture temporal dependencies along the time axis.
        3. FC Layer   — map the final hidden state to 3 output classes
                        (0 = Down, 1 = Stationary, 2 = Up).

    Expected input shape : (Batch, 1, 100, 40)
                            │      │   │    └─ LOB features (bid/ask price+volume × levels)
                            │      │   └────── time steps (sequence length)
                            │      └────────── input channels (grayscale-style single channel)
                            └───────────────── batch size
    """

    def __init__(
        self,
        num_classes: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── 1. CNN Block ───────────────────────────────────────────────────────
        # Goal: learn spatial relationships between adjacent LOB levels.
        #
        # Kernel design strategy
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │  kernel_size=(1, k)  →  only look across the *feature* dimension   │
        # │  stride / padding chosen so the TIME dimension (dim-2) stays = 100 │
        # └─────────────────────────────────────────────────────────────────────┘
        #
        # After each Conv2d the tensor is (B, C_out, 100, W_out).
        # We use padding=(0, k//2) to keep width roughly constant and
        # a kernel height of 1 so the 100 time-steps are NEVER touched.

        self.cnn = nn.Sequential(
            # --- Layer 1 ---
            # in : (B,  1, 100, 40)
            # out: (B, 32, 100, 40)   padding=(0,1) keeps W=40
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            # --- Layer 2 ---
            # in : (B, 32, 100, 40)
            # out: (B, 32, 100, 40)   same padding
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            # --- Layer 3: collapse the feature dimension ---
            # kernel_size=(1, 40) squashes the 40 LOB columns → 1
            # in : (B, 32, 100, 40)
            # out: (B, 32, 100,  1)
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(1, 40)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            nn.Dropout2d(dropout),
        )
        # After CNN: shape is (B, 32, 100, 1)

        # ── 2. LSTM Block ──────────────────────────────────────────────────────
        # LSTM expects (Batch, SeqLen, Features) when batch_first=True.
        # We will reshape (B, 32, 100, 1) → (B, 100, 32) before feeding in.
        #
        # input_size = 32  (CNN output channels, after squeezing the last dim)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,       # (B, T, F) convention
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── 3. Fully-Connected Output ──────────────────────────────────────────
        # We only use the *final* hidden state → shape (B, lstm_hidden)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    # ── Forward Pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 1, 100, 40)

        Returns:
            logits tensor of shape (B, 3)
        """

        # ── CNN ────────────────────────────────────────────────────────────────
        # in : (B,  1, 100, 40)
        out = self.cnn(x)
        # out: (B, 32, 100,  1)

        # ── Reshape for LSTM ───────────────────────────────────────────────────
        # squeeze last spatial dim → (B, 32, 100)
        out = out.squeeze(-1)
        # permute to (B, 100, 32)  i.e. (Batch, SeqLen, Features)
        out = out.permute(0, 2, 1)

        # ── LSTM ───────────────────────────────────────────────────────────────
        # lstm_out : (B, 100, lstm_hidden)  — all hidden states
        # (h_n, c_n): each (num_layers, B, lstm_hidden)
        lstm_out, (h_n, _) = self.lstm(out)

        # Take the hidden state of the *last* LSTM layer at the *last* time step
        # h_n[-1] → (B, lstm_hidden)
        final_hidden = h_n[-1]

        # ── Classifier ─────────────────────────────────────────────────────────
        # in : (B, lstm_hidden)
        # out: (B, 3)
        logits = self.fc(final_hidden)

        return logits   # (B, 3)


class TransformerLOB(nn.Module):
    """
    Transformer-based model for Limit Order Book (LOB) mid-price prediction.
    
    Architecture:
        1. CNN Block      — extract spatial features (same as DeepLOB)
        2. Positional Enc — add absolute time position information
        3. Transformer    — alternative to LSTM for capturing temporal dependencies
        4. FC Layer       — map final representation to 3 output classes
    
    Key difference from DeepLOB:
        - Replaces LSTM with Transformer (self-attention over all timesteps)
        - Uses sinusoidal positional encoding to inject sequence position
        - Can capture long-range dependencies more effectively than LSTM
    
    Expected input shape: (Batch, 1, 100, 40)
    Expected output shape: (Batch, 3)
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_classes (int): Number of output classes (default: 3)
            d_model (int): Dimension of embeddings / attention (default: 32)
            nhead (int): Number of attention heads (default: 4)
            num_layers (int): Number of Transformer encoder layers (default: 2)
            dim_feedforward (int): Hidden dimension in MLP (default: 128)
            dropout (float): Dropout rate (default: 0.1)
        """
        super().__init__()
        
        # ── 1. CNN Block (identical to DeepLOB) ─────────────────────────────────
        # Extract spatial features across LOB levels while preserving time dimension
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(1, 40)),  # collapse feature dimension
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            
            nn.Dropout2d(dropout),
        )
        # After CNN: (B, 32, 100, 1)

        # Project CNN channels (32) to Transformer embedding size (d_model).
        self.input_proj = nn.Linear(32, d_model)
        
        # ── 2. Positional Encoding ─────────────────────────────────────────────
        # Adds sinusoidal positional info so Transformer knows "time order"
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=100, dropout=dropout)
        
        # ── 3. Transformer Encoder ─────────────────────────────────────────────
        # Multi-head self-attention + feed-forward network, repeated num_layers times
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model) convention
            activation='relu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ── 4. Fully-Connected Output ──────────────────────────────────────────
        # Map the final (end-of-sequence) token to class logits
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 1, 100, 40)
        
        Returns:
            logits tensor of shape (B, 3)
        """
        # ── CNN ─────────────────────────────────────────────────────────────────
        # in : (B,  1, 100, 40)
        out = self.cnn(x)
        # out: (B, 32, 100,  1)
        
        # ── Reshape for Transformer ─────────────────────────────────────────────
        # squeeze last spatial dim → (B, 32, 100)
        out = out.squeeze(-1)
        # permute to (B, 100, 32)  i.e. (Batch, SeqLen, d_model)
        out = out.permute(0, 2, 1)
        # project to (B, 100, d_model)
        out = self.input_proj(out)
        
        # ── Positional Encoding ─────────────────────────────────────────────────
        # Add learnable position awareness to the sequence
        out = self.pos_encoder(out)
        
        # ── Transformer Encoder ────────────────────────────────────────────────
        # Self-attention over all timesteps
        # transformer_out: (B, 100, d_model)
        transformer_out = self.transformer_encoder(out)
        
        # Take the LAST timestep → (B, d_model)
        final_out = transformer_out[:, -1, :]
        
        # ── Classifier ──────────────────────────────────────────────────────────
        # in : (B, d_model)
        # out: (B, 3)
        logits = self.fc(final_out)
        
        return logits


# ── Quick Smoke-Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    # Test DeepLOB
    print("=" * 70)
    print("Testing DeepLOB (CNN + LSTM)")
    print("=" * 70)
    model1 = DeepLOB(
        num_classes=3,
        lstm_hidden=64,
        lstm_layers=1,
    ).to(device)

    dummy = torch.randn(32, 1, 100, 40, device=device)
    print(f"Input shape : {dummy.shape}")

    model1.eval()
    with torch.no_grad():
        out1 = model1(dummy)

    print(f"Output shape: {out1.shape}")
    assert out1.shape == (32, 3), f"❌ Unexpected shape: {out1.shape}"
    print("✅ DeepLOB passed.\n")

    total_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"Trainable parameters (DeepLOB): {total_params1:,}\n")

    # Test TransformerLOB
    print("=" * 70)
    print("Testing TransformerLOB (CNN + Transformer)")
    print("=" * 70)
    model2 = TransformerLOB(
        num_classes=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    model2.eval()
    with torch.no_grad():
        out2 = model2(dummy)

    print(f"Output shape: {out2.shape}")
    assert out2.shape == (32, 3), f"❌ Unexpected shape: {out2.shape}"
    print("✅ TransformerLOB passed.\n")

    total_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"Trainable parameters (TransformerLOB): {total_params2:,}\n")

    print("=" * 70)
    print("✅ All model smoke tests passed!")
    print("=" * 70)