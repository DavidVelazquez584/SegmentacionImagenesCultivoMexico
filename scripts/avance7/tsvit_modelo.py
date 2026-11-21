# ==============================================================================
# TSViT (Tarasiou et al., CVPR 2023) — copia de models/TSViT/TSViTdense.py y
# models/TSViT/module.py de github.com/michaeltrs/DeepSatModels (commit 53e6559),
# licencia Apache 2.0. Solo se conservan las clases necesarias para la versión
# "TSViT" de segmentación; el cálculo es idéntico al original.
# ==============================================================================
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    # Única modificación de eficiencia: con USAR_SDPA=True se usa F.scaled_dot_product_attention
    # (kernels fusionados de PyTorch), que calcula softmax(q·kᵀ·escala)·v igual que el original
    # sin materializar la matriz de atención completa.
    USAR_SDPA = True

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        if Attention.USAR_SDPA:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn = dots.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TSViT(nn.Module):
    """Temporal-Spatial ViT (sección 4.3 del artículo). Entrada: B x T x H x W x C, con el
    día del año normalizado (doy / 365.0001) como último canal."""
    def __init__(self, model_config):
        super().__init__()
        self.legacy_reshape = model_config.get('legacy_reshape', False)
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.temporal_depth = model_config.get('temporal_depth', model_config.get('depth'))
        self.spatial_depth = model_config.get('spatial_depth', model_config.get('depth'))
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2   # -1: excluye el canal de fecha
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                             self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.patch_size ** 2))

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim)
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2)
        if self.legacy_reshape:
            # Reconstrucción ORIGINAL (commit 0bc3d10). Los pesos oficiales del Drive se entrenaron
            # con ella; el commit d89f57b la conservó como `legacy_reshape: True` para cargarlos.
            x = x.permute(0, 2, 3, 1).reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)
        else:
            # Reconstrucción corregida (commits d89f57b / a9d20fd): cada parche 2x2 vuelve a su lugar.
            x = rearrange(x, 'b c (h1 w1) (p1 p2) -> b c (h1 p1) (w1 p2)',
                          h1=self.num_patches_1d, w1=self.num_patches_1d, p1=self.patch_size, p2=self.patch_size)
        return x


class TSViTDias(TSViT):
    """Variante para el Avance 7: idéntica a TSViT (reconstrucción espacial corregida) salvo la
    codificación temporal. En lugar del one-hot del día del año (366 valores, que confunde fechas de
    2018 y 2019 con el mismo día del año) usa el one-hot de los días transcurridos desde 2018-09-01,
    la misma referencia temporal que U-TAE. El índice 0 se reserva para las fechas de relleno, igual
    que en el original (ninguna adquisición de PASTIS cae en el día 0).
    Entrada: x (B, T, H, W, 10) normalizada y dias (B, T) enteros."""
    def __init__(self, model_config, n_dias=430):
        super().__init__(dict(model_config, legacy_reshape=False))
        self.n_dias = n_dias
        self.to_temporal_embedding_input = nn.Linear(n_dias, self.dim)

    def forward(self, x, dias):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        xt = F.one_hot(dias.clamp(0, self.n_dias - 1).long(), num_classes=self.n_dias).to(x.dtype)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt.reshape(-1, self.n_dias)).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x = x + temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens.to(x.dtype), x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim)
        x = x + self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2)
        return rearrange(x, 'b c (h1 w1) (p1 p2) -> b c (h1 p1) (w1 p2)',
                         h1=self.num_patches_1d, w1=self.num_patches_1d, p1=self.patch_size, p2=self.patch_size)


# Configuración de configs/PASTIS24/TSViT_fold1.yaml (y TSViT-S: dim 64, dim_head 16)
CONFIG_TSVIT = dict(architecture="TSViT", img_res=24, max_seq_len=60, num_channels=11, num_features=16,
                    num_classes=19, ignore_background=False, dropout=0., patch_size=2, dim=128,
                    temporal_depth=4, spatial_depth=4, heads=4, pool='cls', dim_head=32, emb_dropout=0.,
                    scale_dim=4)
CONFIG_TSVIT_S = dict(CONFIG_TSVIT, dim=64, dim_head=16)

# Normalización fija del repositorio (data/PASTIS24/data_transforms.py, clase Normalize)
MEAN_OFICIAL = [1165.9398193359375, 1375.6534423828125, 1429.2191162109375, 1764.798828125, 2719.273193359375,
                3063.61181640625, 3205.90185546875, 3319.109619140625, 2422.904296875, 1639.370361328125]
STD_OFICIAL = [1942.6156005859375, 1881.9234619140625, 1959.3798828125, 1867.2239990234375, 1754.5850830078125,
               1769.4046630859375, 1784.860595703125, 1767.7100830078125, 1458.963623046875, 1299.2833251953125]
