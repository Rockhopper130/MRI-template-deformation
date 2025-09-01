import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IcosahedralPool(nn.Module):
    def __init__(self, pool_map: torch.LongTensor, mode: str = "avg"):
        super().__init__()
        assert mode in ("avg", "max")
        self.register_buffer("pool_map", pool_map.clone())
        self.mode = mode

    def forward(self, x):
        # x: [B, C, Nf]
        B, C, Nf = x.shape
        pool_map = self.pool_map.to(x.device)
        Nc, k = pool_map.shape
        flat = pool_map.view(-1)
    
        safe_flat = flat.clamp(min=0)    
        gathered = x.index_select(2, safe_flat)  # [B, C, Nc*k]
        gathered = gathered.view(B, C, Nc, k)
        
        mask = (pool_map.unsqueeze(0).unsqueeze(1) != -1).to(x.dtype).to(x.device)
        if self.mode == "avg":
            mask_expand = mask
            summed = (gathered * mask_expand).sum(dim=-1)
            counts = mask_expand.sum(dim=-1).clamp(min=1.0)
            pooled = summed / counts
            return pooled
        else:
            neg_inf = torch.finfo(x.dtype).min / 2
            filled = gathered.clone()
            filled = filled.masked_fill(mask.unsqueeze(0).unsqueeze(1) == 0, neg_inf)
            pooled, _ = filled.max(dim=-1)
            pooled = pooled
            return pooled

class IcosahedralUnpool(nn.Module):
    def __init__(self, up_map: torch.LongTensor):
        super().__init__()
        self.register_buffer("up_map", up_map.clone())

    def forward(self, coarse_feats):
        # coarse_feats: [B, Cc, Nc]
        up_map = self.up_map.to(coarse_feats.device)
        fine_feats = coarse_feats.index_select(2, up_map)
        return fine_feats  # [B, Cc, Nf]

class Conv1x1Block(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, 1), nn.ReLU(inplace=True)]
        if use_bn:
            layers.insert(1, nn.BatchNorm1d(in_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class IcoDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class IcoUpBlock(nn.Module):
    def __init__(self, coarse_ch, skip_ch, out_ch, up_map: torch.LongTensor, use_bn=True):
        super().__init__()
        self.unpool = IcosahedralUnpool(up_map)
        self.net = nn.Sequential(
            nn.Conv1d(coarse_ch + skip_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, coarse_feats, skip_feats):
        up = self.unpool(coarse_feats)
        x = torch.cat([up, skip_feats], dim=1)
        return self.net(x)
    
class IcoUNet(nn.Module):
    def __init__(self, in_ch, channels, pool_maps, up_maps, pool_mode="avg", use_bn=True):
        super().__init__()
        assert len(pool_maps) == len(up_maps)
        levels = len(pool_maps)
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        chs = [in_ch] + channels
        for i in range(levels):
            self.down_blocks.append(IcoDownBlock(chs[i], chs[i+1], use_bn=use_bn))
            self.pools.append(IcosahedralPool(pool_maps[i].to(device), mode=pool_mode))
        self.bottleneck = IcoDownBlock(chs[-1], chs[-1]*2, use_bn=use_bn)
                # --- CORRECTED LOOP ---
        for i in range(levels-1, -1, -1):
            # The coarse_ch for the current level 'i' is the out_ch of the level 'i+1' block.
            # The out_ch of the block at level 'j' is chs[j+1].
            # So, the coarse_ch for level 'i' is chs[i+1+1] = chs[i+2].
            coarse_ch = chs[i+1] * 2 if i == levels - 1 else chs[i+2]
            skip_ch = chs[i+1]
            out_ch = chs[i+1]
            self.up_blocks.append(IcoUpBlock(coarse_ch, skip_ch, out_ch, up_maps[i].to(device), use_bn=use_bn))
            
        # The final_conv input channel should be the output of the last up-block (i=0)
        # The out_ch for the i=0 block is chs[0+1] = chs[1].
        self.final_conv = nn.Conv1d(chs[1], in_ch, 1)

    def forward(self, x):
        # x: [B, C_in, N0]
        skips = []
        cur = x
        for down, pool in zip(self.down_blocks, self.pools):
            cur = down(cur)
            skips.append(cur)
            cur = pool(cur)
        cur = self.bottleneck(cur)
        for up in self.up_blocks:
            skip = skips.pop()
            cur = up(cur, skip)
        out = self.final_conv(cur)
        return out