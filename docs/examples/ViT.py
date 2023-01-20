from torchmix import (
    Add,
    Attach,
    Attention,
    ClassEmbedding,
    ClassPool,
    DropActivation,
    DropAttention,
    DropProjection,
    DropProjectionOut,
    Feedforward,
    PatchEmbedding,
    PositionalEmbedding,
    PreNorm,
    Repeat,
    nn,
)

ViT = nn.Sequential(
    Attach(
        ClassEmbedding(dim=768),
        Add(
            PatchEmbedding(patch_size=16),
            PositionalEmbedding(dim=768, seq_len=196),
        ),
    ),
    Repeat(
        nn.Sequential(
            PreNorm(
                Attention(
                    dim=768,
                    num_heads=12,
                    head_dim=64,
                    plugins=[
                        DropAttention(p=0.1),
                        DropProjection(p=0.1),
                    ],
                ),
                dim=768,
            ),
            PreNorm(
                Feedforward(
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
    ClassPool(0),
    nn.Linear(768, 1000),
)
