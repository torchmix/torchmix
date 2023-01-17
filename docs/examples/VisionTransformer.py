from torchmix import (
    MLP,
    Add,
    Attach,
    Attention,
    DropActivation,
    DropAttention,
    DropProjection,
    DropProjectionOut,
    Extract,
    PatchEmbed,
    PositionEmbed,
    PreNorm,
    Repeat,
    Token,
    nn,
)

ViT = nn.Sequential(
    Attach(
        Token(dim=768),
        Add(
            PatchEmbed(patch_size=16),
            PositionEmbed(seq_length=196, dim=768),
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
    Extract(0),
    nn.Linear(768, 1000),
)