from torchmix import (
    MLP,
    Attention,
    DropActivation,
    DropAttention,
    DropProjection,
    DropProjectionOut,
    PreNorm,
    RelativePositionBias,
    Repeat,
    nn,
)

BERT = nn.Sequential(
    nn.Embedding(50257, 768),
    Repeat(
        nn.Sequential(
            PreNorm(
                Attention(
                    dim=768,
                    num_heads=12,
                    head_dim=64,
                    plugins=[
                        RelativePositionBias(
                            seq_length=1024,
                            num_buckets=256,
                            num_heads=12,
                        ),
                        DropAttention(p=0.1),
                        DropProjection(p=0.1),
                    ],
                ),
                dim=768,
            ),
            PreNorm(
                MLP(
                    dim=768,
                    act_layer=nn.GELU(),
                    expansion_factor=4,
                    plugins=[
                        DropActivation(p=0.1),
                        DropProjectionOut(p=0.1),
                    ],
                ),
                dim=768,
            ),
        ),
        depth=12,
    ),
    nn.Linear(768, 50257),
)