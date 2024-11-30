from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block, Attention, DropPath, Mlp
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class CrossAttention(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Cross-Attention block where image embeddings cross-attend to text embeddings.
        Args:
            img_dim (int): Hidden dimension of image embeddings.
            text_dim (int): Hidden dimension of text embeddings.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, adds bias to QKV projections.
            qk_scale (float or None): Scaling factor for QK attention scores. Defaults to 1/sqrt(head_dim).
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        self.img_dim = img_dim
        self.text_dim = text_dim

        # Ensure dimensions are divisible by the number of heads
        assert img_dim % num_heads == 0, "Image dimension must be divisible by the number of heads"
        assert text_dim % num_heads == 0, "Text dimension must be divisible by the number of heads"

        self.img_head_dim = img_dim // num_heads
        self.text_head_dim = text_dim // num_heads
        
        # Scaling factor for attention
        self.scale = qk_scale or self.text_head_dim ** -0.5

        # Q projection from image embeddings
        self.q_proj = nn.Linear(img_dim, img_dim, bias=qkv_bias)
        
        # K and V projections from text embeddings
        self.k_proj = nn.Linear(text_dim, img_dim, bias=qkv_bias)  # Project text to match image dim
        self.v_proj = nn.Linear(text_dim, img_dim, bias=qkv_bias)  # Project text to match image dim

        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection for image embeddings
        self.proj = nn.Linear(img_dim, img_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, img_embeds, text_embeds, text_attention_mask):
        """
        Forward pass for cross-attention.
        
        Args:
            img_embeds (torch.Tensor): Image embeddings of shape (B, N_img, img_dim).
            text_embeds (torch.Tensor): Text embeddings of shape (B, N_text, text_dim).
            text_attention_mask (torch.Tensor): Attention mask for text embeddings of shape (B, N_text).
        
        Returns:
            torch.Tensor: Updated image embeddings after cross-attending to text embeddings.
                          Shape: (B, N_img, img_dim).
        """
        B, N_img, _ = img_embeds.shape
        _, N_text, _ = text_embeds.shape

        # Compute Q from image embeddings
        q = self.q_proj(img_embeds).reshape(B, N_img, self.num_heads, self.img_head_dim).permute(0, 2, 1, 3)

        # Compute K and V from text embeddings
        k = self.k_proj(text_embeds).reshape(B, N_text, self.num_heads, self.img_head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(text_embeds).reshape(B, N_text, self.num_heads, self.img_head_dim).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention: Attention(QK^T) * V
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # Shape: (B, num_heads, N_img, N_text)

        # Apply the mask: set scores for padding tokens to a large negative value
        mask_expanded = text_attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, N_text)
        attn_scores = attn_scores.masked_fill(~mask_expanded.bool(), float('-inf'))

        attn_probs = attn_scores.softmax(dim=-1)              # Normalize across the last dimension (N_text)
        attn_probs = self.attn_drop(attn_probs)

        # Apply attention to V
        attended_values = (attn_probs @ v)                    # Shape: (B, num_heads, N_img, head_dim)
        
        # Reshape back to original dimensions
        attended_values = attended_values.permute(0, 2, 1, 3).reshape(B, N_img, -1)

        # Project the attended values back to the original image embedding space
        output = self.proj(attended_values)
        output = self.proj_drop(output)

        return output

class TextConditionedBlock(nn.Module):
    def __init__(self, dim, text_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Self attention branch
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # Cross attention branch
        self.norm_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            img_dim=dim, text_dim=text_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # FFN branch
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, text_embeddings, text_attention_mask, skip_cross_attn):
        # Self attention
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        # Cross attention
        if not skip_cross_attn:
            x = x + self.drop_path(self.cross_attn(self.norm_cross(x), text_embeddings, text_attention_mask))
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TextConditionedMaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 text_dim=1024):  # Added text_dim parameter
        super().__init__()

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), 
                                    requires_grad=False)

        # Modified to use TextConditionedBlock
        self.blocks = nn.ModuleList([
            TextConditionedBlock(embed_dim, text_dim, num_heads, mlp_ratio,
                               qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # ITM head
        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 
                                            decoder_embed_dim), requires_grad=False)

        # Modified decoder blocks to use TextConditionedBlock
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, 
                                    bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, text_embeddings, text_attention_mask, mask_ratio, skip_cross_attn):
        # embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, text_embeddings, text_attention_mask, skip_cross_attn)
        x = self.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], 
                                           ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, 
                         index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        
        return x

    def forward(self, imgs, text_embeddings, text_attention_mask, mask_ratio=0.75, skip_cross_attn=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, text_embeddings, text_attention_mask, mask_ratio, skip_cross_attn)
        cls_tokens = latent[:, :1, :]
        itm_probs = self.itm_head(cls_tokens)
        pred = self.forward_decoder(latent, ids_restore)
        reconstruction_loss = self.forward_loss(imgs, pred, mask)
        return reconstruction_loss, pred, mask, itm_probs

    # Keep all other methods from original MAE unchanged
    def patchify(self, imgs):
        """Same as original MAE"""
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Same as original MAE"""
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Same as original MAE"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, 
                              index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """Same as original MAE"""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


def copy_weights_mae_textmae(mae_model, text_conditioned_model):
    # Copy encoder weights
    text_conditioned_model.patch_embed.load_state_dict(mae_model.patch_embed.state_dict())
    text_conditioned_model.cls_token.data.copy_(mae_model.cls_token.data)
    text_conditioned_model.pos_embed.data.copy_(mae_model.pos_embed.data)
    
    # Copy encoder blocks (ignoring cross-attention specific parameters)
    for mae_block, text_block in zip(mae_model.blocks, text_conditioned_model.blocks):
        # Copy self-attention and MLP weights
        text_block.self_attn.load_state_dict(mae_block.attn.state_dict())
        text_block.mlp.load_state_dict(mae_block.mlp.state_dict())
        # Copy norm layers
        text_block.norm1.load_state_dict(mae_block.norm1.state_dict())
        text_block.norm2.load_state_dict(mae_block.norm2.state_dict())

    text_conditioned_model.norm.load_state_dict(mae_model.norm.state_dict())

    # Copy decoder weights
    text_conditioned_model.decoder_embed.load_state_dict(mae_model.decoder_embed.state_dict())
    text_conditioned_model.mask_token.data.copy_(mae_model.mask_token.data)
    text_conditioned_model.decoder_pos_embed.data.copy_(mae_model.decoder_pos_embed.data)

    for mae_decoder_block, text_decoder_block in zip(mae_model.decoder_blocks, text_conditioned_model.decoder_blocks):
        # Assuming decoder blocks are not using cross-attention
        text_decoder_block.load_state_dict(mae_decoder_block.state_dict())

    text_conditioned_model.decoder_norm.load_state_dict(mae_model.decoder_norm.state_dict())
    text_conditioned_model.decoder_pred.load_state_dict(mae_model.decoder_pred.state_dict())

    return text_conditioned_model


def get_mae_vit_large_patch16_text_conditioned(**kwargs):
    model = TextConditionedMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, text_dim=1024, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model