
import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Compatibility helpers
# ----------------------------------------------------------------------------
def randn_like_compat(x: torch.Tensor, generator=None):
    """torch.randn_like() in some torch builds doesn't accept `generator=`.
    Use this to keep deterministic behavior when a Generator is provided."""
    if generator is None:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
# SD3
from diffusers import StableDiffusion3Pipeline

from .flowedit_sd3 import generate_scout_image_sd3, load_sd3_pipeline
from .project_utils import (
    case_scout_path,
    case_output_path,
    filter_records,
    iter_cases,
    load_dataset,
    load_yaml,
    prepare_case_dirs,
    save_case_manifest,
)


# =============================================================================
# IAM-Flow (SD3-medium backbone)
# - 淇濇寔浣?v12 鐗?IAM-Flow 鐨勬暣浣撶瓥鐣ュ畬鍏ㄤ竴鑷达細
#   * FlowEdit 螖v 宸垎椹卞姩
#   * 鍔ㄦ€侀槇鍊?纭害閫€鐏?+ Gaussian blur
#   * core / ring / outer 涓夊垎鍖?#   * slack銆乺ing銆乤nchor_rho銆佺紦瀛?inversion path
# - 鍞竴鍏抽敭宸紓锛歮ask 鎻愬彇鏀逛负 SD3 joint attention
#   (鍙傝€冧綘鎻愪緵鐨?SD3 vs FLUX mask 鏂囨。)
# =============================================================================

# =============================================================================
# 1) Attention Hook锛氬湪 PyTorch SDPA 灞傞潰鎶?joint attention
#    - SD3 joint seq: [text_tokens (N_T), image_tokens (N_I)]
#    - 闇€瑕佸悓鏃舵娊鍙栵細
#        Image -> Text (A_IT)
#        Text  -> Image (A_TI)
#      骞跺绉板钩鍧?# =============================================================================

attn_store = []
_original_sdpa = F.scaled_dot_product_attention


def _safe_float(x):
    # SDPA hook 鍐呴儴璁＄畻鐢?float32 鏇寸ǔ
    return x.float() if x.dtype != torch.float32 else x


def get_hooked_sdpa_sd3(
    L_img: int,
    w_txt: torch.Tensor,
    attn_reduce: str = "sym",   # {"sym","it","ti","max"}
    norm_mode: str = "full",    # {"full","cross"}; "cross" = cross-only softmax
):
    """
    L_img: image token 鏁?(= H_p * W_p)
    w_txt: (B, N_T) 鏂囨湰 token 鏉冮噸锛堝凡缁忓畬鎴?CLIP/T5 鍔犳潈涓?offset锛?
    娉ㄦ剰锛歋D3 joint attention 鐨勫簭鍒楀竷灞€鍦ㄦ湰鑴氭湰閲屽亣瀹氫负 [Image, Text]銆?    """

    attn_reduce = (attn_reduce or "sym").lower()
    norm_mode = (norm_mode or "full").lower()

    def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        # 鍏堟甯歌窇 attention
        out = _original_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs
        )

        L_seq = query.shape[-2]
        if L_seq <= L_img:
            return out  # 涓嶆槸 joint (text+image) attention

        L_txt = L_seq - L_img

        with torch.no_grad():
            q = _safe_float(query)
            k = _safe_float(key)

            B = q.shape[0]
            if w_txt.shape[0] != B:
                if w_txt.shape[0] == 1:
                    w = w_txt.expand(B, -1)
                else:
                    w = w_txt[:B]
            else:
                w = w_txt

            # pad/trunc to L_txt
            if w.shape[1] > L_txt:
                w = w[:, :L_txt]
            elif w.shape[1] < L_txt:
                pad = torch.zeros((w.shape[0], L_txt - w.shape[1]), device=w.device, dtype=w.dtype)
                w = torch.cat([w, pad], dim=1)

            w_sum = torch.clamp(w.sum(dim=1, keepdim=True), min=1e-6)

            # SD3 joint layout: [Image, Text]
            q_img = q[:, :, :L_img, :]   # (B, heads, L_img, d)
            q_txt = q[:, :, L_img:, :]   # (B, heads, L_txt, d)
            k_all = k                    # (B, heads, L_seq, d)

            scale_factor = scale or (1.0 / (query.shape[-1] ** 0.5))

            # ---- Image -> (Image+Text) ----
            score_img = torch.matmul(q_img, k_all.transpose(-2, -1)) * scale_factor
            if attn_mask is not None:
                score_img = score_img + _safe_float(attn_mask[:, :, :L_img, :])

            if norm_mode == "cross":
                score_img_cross = score_img[:, :, :, L_img:]  # (B, heads, L_img, L_txt)
                A_IT = torch.softmax(score_img_cross, dim=-1)
            else:
                prob_img = torch.softmax(score_img, dim=-1)
                A_IT = prob_img[:, :, :, L_img:]  # (B, heads, L_img, L_txt)

            A_IT = A_IT.mean(dim=1)  # (B, L_img, L_txt)
            it_vec = (A_IT * w.unsqueeze(1)).sum(dim=-1) / w_sum  # (B, L_img)

            # ---- Text -> (Image+Text) ----
            score_txt = torch.matmul(q_txt, k_all.transpose(-2, -1)) * scale_factor
            if attn_mask is not None:
                score_txt = score_txt + _safe_float(attn_mask[:, :, L_img:, :])

            if norm_mode == "cross":
                score_txt_cross = score_txt[:, :, :, :L_img]  # (B, heads, L_txt, L_img)
                A_TI = torch.softmax(score_txt_cross, dim=-1)
            else:
                prob_txt = torch.softmax(score_txt, dim=-1)
                A_TI = prob_txt[:, :, :, :L_img]  # (B, heads, L_txt, L_img)

            A_TI = A_TI.mean(dim=1)  # (B, L_txt, L_img)
            ti_vec = (A_TI * w.unsqueeze(-1)).sum(dim=1) / w_sum  # (B, L_img)

            if attn_reduce == "it":
                vec = it_vec
            elif attn_reduce == "ti":
                vec = ti_vec
            elif attn_reduce == "max":
                vec = torch.maximum(it_vec, ti_vec)
            else:
                # default: symmetric average
                vec = 0.5 * (it_vec + ti_vec)

            attn_store.append(vec.detach().cpu())

        return out

    return hooked_sdpa


# =============================================================================
# 2) Token span 瑙ｆ瀽锛堜笁濂?tokenizer锛歝lip1/clip2/t5锛?# =============================================================================

def get_token_indices(tokenizer, prompt: str, tokens_to_search, max_length: int = None, match_mode: str = "string"):
    """
    鍦ㄧ粰瀹?tokenizer 鐨?token 搴忓垪涓紝瀹氫綅 tokens_to_search 瀵瑰簲鐨?token indices锛堝惈瀛愯瘝锛夈€?
    match_mode:
      - "string": 瀹芥澗瀛楃涓插尮閰嶏紙浣犲師 SD3 鑴氭湰鐨勭瓥鐣ワ級
      - "ids":    鐢?input_ids 瀛愬簭鍒楀尮閰嶏紙鏇寸ǔ锛屽挨鍏跺澶?token / sentencepiece / byte-bpe锛?    """
    if not tokens_to_search:
        return []

    match_mode = (match_mode or "string").lower()
    max_length = int(max_length) if max_length is not None else int(getattr(tokenizer, "model_max_length", 77))

    encoding = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding.input_ids[0].tolist()

    # 鏈変簺 tokenizer 浼氭妸 pad 鏀惧湪鏈熬锛涘尮閰嶆椂閬垮厤璺ㄥ埌 pad 鍖哄煙
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None and pad_id in input_ids:
        effective_len = input_ids.index(pad_id)
        prompt_ids = input_ids[:effective_len]
    else:
        prompt_ids = input_ids

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if pad_id is not None:
        special_ids.add(pad_id)

    if match_mode == "ids":
        indices = []
        for phrase in tokens_to_search:
            if not phrase:
                continue

            enc_p = tokenizer(
                phrase.replace("_", " "),
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            sub_ids = [int(x) for x in enc_p.input_ids[0].tolist() if int(x) not in special_ids]
            if len(sub_ids) == 0:
                continue

            # 绾挎€у瓙搴忓垪鎼滅储
            Lp = len(prompt_ids)
            Ls = len(sub_ids)
            for s in range(0, max(Lp - Ls + 1, 0)):
                if prompt_ids[s:s+Ls] == sub_ids:
                    indices.extend(list(range(s, s + Ls)))

        return sorted(list(set(indices)))

    # -------------------------
    # fallback: 鍘熸潵鐨勫鏉惧瓧绗︿覆鍖归厤
    # -------------------------
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    indices = []
    for t_search in tokens_to_search:
        words = t_search.replace("_", " ").split()
        for word in words:
            clean_search = word.lower().strip()
            for i, t in enumerate(tokens):
                if hasattr(tokenizer, "all_special_tokens") and t in tokenizer.all_special_tokens:
                    continue
                clean_t = (t.lower()
                    .replace(" ", "")
                    .replace("\u2581", "")  # sentencepiece marker
                    .replace("臓", "").replace("摹", "")  # byte-bpe word boundary
                    .replace("</w>", "")
                    .replace("##", "")
                    .strip())
                if not clean_t:
                    continue
                if (
                    clean_t == clean_search
                    or (clean_t.startswith(clean_search) and len(clean_search) >= 3)
                    or (clean_search in clean_t and len(clean_t) <= len(clean_search) + 2)
                ):
                    indices.append(i)

    return sorted(list(set(indices)))


def build_joint_text_weights(
    pipe,
    prompt: str,
    tokens_to_search,
    w_clip: float = 1.0,
    w_t5: float = 0.3,
    max_t5_length: int = 256,
    token_match_mode: str = "string",  # {"string","ids"}
    device: str = "cuda",
):
    """
    鉁?SD3 姝ｇ‘鐨?text token 甯冨眬锛堜笌 diffusers 鐨?encode_prompt 瀵归綈锛夛細

    - 涓や釜 CLIP Text Encoder 鐨?*token 搴忓垪闀垮害鐩稿悓*锛堥€氬父 77锛夛紝骞朵笖鍦?encode_prompt 閲屾槸锛?        clip_prompt_embeds = cat([clip1_embed, clip2_embed], dim=-1)   # 鐗瑰緛缁存嫾鎺ワ紙涓嶆槸 token 缁达紒锛?    - 鐒跺悗鎶?CLIP 鐨?token 搴忓垪 (77) 鍜?T5 鐨?token 搴忓垪 (max_sequence_length) 鍦?token 缁存嫾鎺ワ細
        prompt_embeds = cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    鍥犳 joint 鐨?text 搴忓垪鏄細
        [CLIP tokens (N_clip=77), T5 tokens (N_t5=max_t5_length)]
    鑰屼笉鏄?[clip1, clip2, t5] 涓夋 token 鎷兼帴銆?
    鏈嚱鏁拌緭鍑?w_joint: (N_clip + N_t5,) 鐨勬潈閲嶅悜閲忥紝渚?joint attention 閲?text 娈靛姞鏉冧娇鐢ㄣ€?    """
    tok1 = getattr(pipe, "tokenizer", None)
    tok2 = getattr(pipe, "tokenizer_2", None)
    tok3 = getattr(pipe, "tokenizer_3", None)
    if tok1 is None or tok2 is None or tok3 is None:
        raise AttributeError("SD3 pipeline requires tokenizer, tokenizer_2, and tokenizer_3.")

    N_clip = int(getattr(pipe, "tokenizer_max_length", getattr(tok1, "model_max_length", 77)))

    # 杩欎袱濂?CLIP tokenizer 閮戒細 pad/trunc 鍒?N_clip
    _ = tok1(prompt, padding="max_length", truncation=True, max_length=N_clip, return_tensors="pt")
    _ = tok2(prompt, padding="max_length", truncation=True, max_length=N_clip, return_tensors="pt")

    enc3 = tok3(prompt, padding="max_length", truncation=True, max_length=max_t5_length, return_tensors="pt")
    N_t5 = int(enc3.input_ids.shape[1])

    idx1 = get_token_indices(tok1, prompt, tokens_to_search, max_length=N_clip, match_mode=token_match_mode)
    idx2 = get_token_indices(tok2, prompt, tokens_to_search, max_length=N_clip, match_mode=token_match_mode)
    idx3 = get_token_indices(tok3, prompt, tokens_to_search, max_length=max_t5_length, match_mode=token_match_mode)

    idx_clip = sorted(list(set([i for i in idx1 if 0 <= i < N_clip] + [i for i in idx2 if 0 <= i < N_clip])))
    idx_t5 = sorted([i for i in idx3 if 0 <= i < N_t5])

    # joint weights: [clip(77), t5(max_t5_length)]
    w = torch.zeros((N_clip + N_t5,), device=device, dtype=torch.float32)
    for j in idx_clip:
        w[j] = w_clip
    for j in idx_t5:
        w[N_clip + j] = w_t5

    meta = {
        "N_clip": int(N_clip),
        "N_t5": int(N_t5),
        "N_T": int(N_clip + N_t5),
        "idx_clip1": idx1,
        "idx_clip2": idx2,
        "idx_clip_union": idx_clip,
        "idx_t5": idx_t5,
        "token_match_mode": str(token_match_mode),
        "idx_joint_nonzero": (w > 0).nonzero(as_tuple=False).squeeze(-1).tolist(),
    }
    return w, meta
# =============================================================================
# 3) Mask 鏋勯€犲伐鍏峰嚱鏁帮紙涓?v12 淇濇寔涓€鑷达級
# =============================================================================

def min_max_norm(t_map: torch.Tensor):
    v_min, v_max = t_map.min(), t_map.max()
    if (v_max - v_min) > 1e-6:
        return (t_map - v_min) / (v_max - v_min)
    return torch.zeros_like(t_map)


def run_attention_scout_sd3(
    pipe,
    scout_latents,
    scout_prompt_embeds,
    scout_pooled_embeds,
    scout_w_txt,
    timesteps,
    L_img,
    selected_layers,
    attn_reduce: str = "sym",
    norm_mode: str = "full",
):
    """
    鐢?SDPA hook 鎶撳彇 joint attention锛屽苟寰楀埌 (B, L_img) 鐨?token-weighted 鍝嶅簲銆?    鐒跺悗閫夊眰 (8-17) 骞冲潎锛岃緭鍑?(B, L_img)銆?    """
    attn_store.clear()
    F.scaled_dot_product_attention = get_hooked_sdpa_sd3(L_img=L_img, w_txt=scout_w_txt, attn_reduce=attn_reduce, norm_mode=norm_mode)

    try:
        with torch.no_grad():
            _ = transformer_forward(
                pipe=pipe,
                latents=scout_latents,
                timestep=timesteps,
                encoder_hidden_states=scout_prompt_embeds,
                pooled_projections=scout_pooled_embeds,
            )
    finally:
        F.scaled_dot_product_attention = _original_sdpa

    if len(attn_store) == 0:
        raise RuntimeError("Attention hook did not capture any SD3 joint-attention outputs.")

    valid_layers = [j for j in selected_layers if 0 <= j < len(attn_store)]
    if len(valid_layers) == 0:
        raise RuntimeError(
            f"Captured {len(attn_store)} attention tensors, which is insufficient for {selected_layers}."
        )

    A_layers = torch.stack([attn_store[j] for j in valid_layers], dim=0)  # (L, B, L_img)
    A = A_layers.mean(dim=0)  # (B, L_img)
    attn_store.clear()
    return A


def extract_spatial_map(vec_Limg: torch.Tensor, H_p: int, W_p: int, blur_sigma: float, device):
    """
    vec_Limg: (L_img,) or (B, L_img)
    杈撳嚭锛?(H_p, W_p)锛堜粎鍙?batch=0锛夋垨 (B,H_p,W_p)
    """
    if vec_Limg.dim() == 1:
        M = vec_Limg.view(H_p, W_p).float()
        M = TF.gaussian_blur(M.unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=blur_sigma).squeeze()
        return min_max_norm(M).to(device)
    else:
        B = vec_Limg.shape[0]
        M = vec_Limg.view(B, H_p, W_p).float()
        # 鎵瑰鐞?blur
        M_blur = []
        for b in range(B):
            Mb = TF.gaussian_blur(M[b].unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=blur_sigma).squeeze()
            M_blur.append(min_max_norm(Mb))
        return torch.stack(M_blur, dim=0).to(device)


def build_union_edit_map_sd3(
    pipe,
    x_src,
    x_tgt,
    scout_noise,
    src_prompt_embeds,
    tgt_prompt_embeds,
    src_pooled,
    tgt_pooled,
    w_src,
    w_tgt,
    timestep,
    sigma_t,
    L_img,
    H_p,
    W_p,
    blur_sigma,
    selected_layers,
    attn_reduce: str = "sym",
    norm_mode: str = "full",
):
    noisy_src = (1 - sigma_t) * x_src + sigma_t * scout_noise
    noisy_tgt = (1 - sigma_t) * x_tgt + sigma_t * scout_noise

    scout_latents = torch.cat([noisy_src, noisy_tgt], dim=0)
    scout_embeds = torch.cat([src_prompt_embeds, tgt_prompt_embeds], dim=0)
    scout_pooled = torch.cat([src_pooled, tgt_pooled], dim=0)

    scout_w = torch.stack([w_src, w_tgt], dim=0).to(device=x_src.device)

    A = run_attention_scout_sd3(
        pipe=pipe,
        scout_latents=scout_latents,
        scout_prompt_embeds=scout_embeds,
        scout_pooled_embeds=scout_pooled,
        scout_w_txt=scout_w,
        timesteps=timestep,
        L_img=L_img,
        selected_layers=selected_layers,
        attn_reduce=attn_reduce,
        norm_mode=norm_mode,
    )  # (2, L_img)

    M_maps = extract_spatial_map(A, H_p, W_p, blur_sigma, x_src.device)  # (2,H_p,W_p)
    M_src, M_tgt = M_maps[0], M_maps[1]
    M = torch.max(M_src, M_tgt)

    debug_maps = {
        "mode": "union",
        "src_token_map": M_src,
        "tgt_token_map": M_tgt,
        "base_map": M,
    }
    return M, debug_maps


def build_unchanged_complement_edit_map_sd3(
    pipe,
    x_src,
    scout_noise,
    src_prompt_embeds,
    src_pooled,
    w_keep,
    timestep,
    sigma_t,
    L_img,
    H_p,
    W_p,
    blur_sigma,
    selected_layers,
    attn_reduce: str = "sym",
    norm_mode: str = "full",
):
    noisy_src = (1 - sigma_t) * x_src + sigma_t * scout_noise

    A = run_attention_scout_sd3(
        pipe=pipe,
        scout_latents=noisy_src,
        scout_prompt_embeds=src_prompt_embeds,
        scout_pooled_embeds=src_pooled,
        scout_w_txt=w_keep.unsqueeze(0).to(device=x_src.device),
        timesteps=timestep,
        L_img=L_img,
        selected_layers=selected_layers,
        attn_reduce=attn_reduce,
        norm_mode=norm_mode,
    )  # (1, L_img)

    M_keep = extract_spatial_map(A[0], H_p, W_p, blur_sigma, x_src.device)
    M = 1.0 - M_keep
    M = torch.clamp(M, 0.0, 1.0)

    debug_maps = {
        "mode": "unchanged_complement",
        "keep_token_map": M_keep,
        "base_map": M,
    }
    return M, debug_maps


def save_debug_maps(debug_maps, out_dir, step_idx):
    mode = debug_maps.get("mode", "unknown")

    if mode == "union":
        Image.fromarray((debug_maps["src_token_map"].detach().cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_dir, f"src_token_map_step_{step_idx:02d}.png")
        )
        Image.fromarray((debug_maps["tgt_token_map"].detach().cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_dir, f"tgt_token_map_step_{step_idx:02d}.png")
        )
        Image.fromarray((debug_maps["base_map"].detach().cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_dir, f"union_base_map_step_{step_idx:02d}.png")
        )

    elif mode == "unchanged_complement":
        Image.fromarray((debug_maps["keep_token_map"].detach().cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_dir, f"unchanged_keep_map_step_{step_idx:02d}.png")
        )
        Image.fromarray((debug_maps["base_map"].detach().cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_dir, f"complement_base_map_step_{step_idx:02d}.png")
        )


# =============================================================================
# 4) SD3 Transformer 鍓嶅悜灏佽 + CFG
# =============================================================================

def transformer_forward(pipe, latents, timestep, encoder_hidden_states, pooled_projections):
    """
    鍏煎涓嶅悓 diffusers 鐗堟湰 SD3 transformer 鐨?forward 鍙傛暟鍛藉悕宸紓銆?    甯歌涓ょ锛?      - sample=..., timestep=...
      - hidden_states=..., timestep=...

    鍚屾椂鍋?timestep 褰㈢姸瀵归綈锛氱‘淇濇槸 (B,) 鐨?tensor銆?    """
    transformer = pipe.transformer

    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep, device=latents.device)
    if timestep.ndim == 0:
        timestep = timestep[None]
    if timestep.shape[0] != latents.shape[0]:
        timestep = timestep.expand(latents.shape[0])

    kwargs = dict(
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False,
    )

    # 鐗堟湰鍏煎锛氫紭鍏?sample锛屽惁鍒?hidden_states
    try:
        return transformer(sample=latents, **kwargs)[0]
    except TypeError:
        return transformer(hidden_states=latents, **kwargs)[0]


def calc_v_sd3(
    pipe,
    latents,
    timestep,
    cond_embeds,
    cond_pooled,
    uncond_embeds,
    uncond_pooled,
    guidance_scale: float,
):
    """
    SD3 涓婄殑 鈥渧elocity鈥濓細
    - 浣犵殑 IAM-Flow / FlowEdit 鍐欐硶鏈熸湜妯″瀷杈撳嚭鍙綋浣?v(t,z)
    - SD3-medium 鍦?diffusers 涓€氬父閰嶅悎 flow-matching scheduler 浣跨敤
    杩欓噷浣跨敤鏍囧噯 CFG锛?      v = v_uncond + s * (v_cond - v_uncond)
    """
    with torch.no_grad():
        if guidance_scale is None:
            return transformer_forward(pipe, latents, timestep, cond_embeds, cond_pooled)

        v_u = transformer_forward(pipe, latents, timestep, uncond_embeds, uncond_pooled)
        v_c = transformer_forward(pipe, latents, timestep, cond_embeds, cond_pooled)
        return v_u + guidance_scale * (v_c - v_u)


def encode_prompt_pair(pipe, prompt: str, device: str, max_t5_length: int = 256):
    """
    鑾峰彇 cond / uncond 鐨?prompt_embeds 涓?pooled_embeds銆?    鍏煎涓嶅悓 diffusers 鐗堟湰鐨?StableDiffusion3Pipeline.encode_prompt锛圫D3 闇€瑕?prompt_2 / prompt_3锛夈€?
    缁熶竴杩斿洖椤哄簭锛堜緵涓嬫父浣跨敤锛夛細
        cond_embeds, pooled_cond, uncond_embeds, pooled_uncond
    """
    import inspect

    neg = ""

    call_kwargs = dict(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=neg,
        negative_prompt_2=neg,
        negative_prompt_3=neg,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        max_sequence_length=max_t5_length,
    )

    sig = inspect.signature(pipe.encode_prompt)
    call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

    out = pipe.encode_prompt(**call_kwargs)

    if not isinstance(out, (tuple, list)):
        raise RuntimeError(f"pipe.encode_prompt returned an unexpected type: {type(out)}")

    if len(out) < 4:
        raise RuntimeError(f"pipe.encode_prompt returned {len(out)} values, which is not enough to parse.")

    a, b, c, d = out[:4]

    # 甯歌涓ょ椤哄簭锛?    # 1) diffusers 涓荤嚎锛圫D3锛?
    #    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) -> dims 3,3,2,2
    # 2) 鏌愪簺鍙樹綋锛?    #    (prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds) -> dims 3,2,3,2
    if hasattr(a, "ndim") and hasattr(b, "ndim") and hasattr(c, "ndim") and hasattr(d, "ndim"):
        if a.ndim == 3 and b.ndim == 3 and c.ndim == 2 and d.ndim == 2:
            cond_embeds, uncond_embeds, pooled_cond, pooled_uncond = a, b, c, d
            return cond_embeds, pooled_cond, uncond_embeds, pooled_uncond
        if a.ndim == 3 and b.ndim == 2 and c.ndim == 3 and d.ndim == 2:
            cond_embeds, pooled_cond, uncond_embeds, pooled_uncond = a, b, c, d
            return cond_embeds, pooled_cond, uncond_embeds, pooled_uncond

    # 鍏滃簳锛氭寜涓荤嚎椤哄簭瑙ｆ瀽
    cond_embeds, uncond_embeds, pooled_cond, pooled_uncond = a, b, c, d
    return cond_embeds, pooled_cond, uncond_embeds, pooled_uncond


# =============================================================================
# 5) Source inversion cache锛堝畬鍏ㄤ繚鐣欎綘 v12 鐨勨€滃姩鎬佸弽婕旈敋瀹氣€濇€濇兂锛?# =============================================================================

# =============================================================================
# 6) 涓荤▼搴忥細IAM-Flow SD3
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # -------------------------
    # 瓒呭弬鏁帮紙淇濇寔浣?v12 椋庢牸锛?    # -------------------------
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    T_steps = 50
    n_avg = 8
    src_guidance_scale = 1.5
    tgt_guidance_scale = 13.5
    n_min = 0
    n_max = 33
    ref_noise_seed = 999

    # mask blur / threshold / sigmoid / ring / slack锛堜笌浣?v12 淇濇寔涓€鑷达級
    blur_sigma_start = 2.0
    blur_sigma_end = 0.6
    q_start = 0.40
    q_end = 0.80
    alpha_min = 5.0
    alpha_max = 20.0
    tau_core = 0.50
    ring_radius_start = 8
    ring_radius_end = 3
    w_ring = 0.5
    slack = 0.05
    anchor_rho = 1.0
    ring_blend = 0.5  # ring 鍖哄仛鏇翠繚瀹堢殑 no-slack 杩囨浮锛屼笉鍐嶄娇鐢ㄥ姩鎬佸弽婕旈敋瀹?
def run_sam_flow_sd3_case(
    pipe: StableDiffusion3Pipeline,
    case: Dict,
    config: Dict,
    scout_image_path: str | Path,
    case_dir: str | Path,
) -> Path:
    sam_cfg = config["sam_flow"]
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    save_maps = bool(config.get("run", {}).get("save_debug_maps", False))

    device = next(pipe.transformer.parameters()).device
    dtype = next(pipe.transformer.parameters()).dtype

    T_steps = int(sam_cfg["T_steps"])
    n_avg = int(sam_cfg["n_avg"])
    src_guidance_scale = float(sam_cfg["src_guidance_scale"])
    tgt_guidance_scale = float(sam_cfg["tgt_guidance_scale"])
    n_min = int(sam_cfg["n_min"])
    n_max = int(sam_cfg["n_max"])
    ref_noise_seed = int(sam_cfg["ref_noise_seed"])

    blur_sigma_start = float(sam_cfg["blur_sigma_start"])
    blur_sigma_end = float(sam_cfg["blur_sigma_end"])
    q_start = float(sam_cfg["q_start"])
    q_end = float(sam_cfg["q_end"])
    alpha_min = float(sam_cfg["alpha_min"])
    alpha_max = float(sam_cfg["alpha_max"])
    tau_core = float(sam_cfg["tau_core"])
    ring_radius_start = int(sam_cfg["ring_radius_start"])
    ring_radius_end = int(sam_cfg["ring_radius_end"])
    w_ring = float(sam_cfg["w_ring"])
    slack = float(sam_cfg["slack"])
    anchor_rho = float(sam_cfg["anchor_rho"])
    ring_blend = float(sam_cfg.get("ring_blend", 0.5))
    w_clip = float(sam_cfg["w_clip"])
    w_t5 = float(sam_cfg["w_t5"])
    max_t5_length = int(sam_cfg["max_t5_length"])
    mask_extract_cfg = dict(sam_cfg.get("mask_extract", {}))
    selected_layers = list(range(int(mask_extract_cfg["layer_start"]), int(mask_extract_cfg["layer_end"])))

    source_prompt = case["source_prompt"]
    target_prompt = case["target_prompt"]
    source_mask_tokens = case["source_mask_tokens"]
    target_mask_tokens = case["target_mask_tokens"]
    unchanged_tokens = case["unchanged_tokens"]

    init_image_src = Image.open(case["image_path"]).convert("RGB")
    if not Path(scout_image_path).exists():
        raise FileNotFoundError(f"Scout image not found: {scout_image_path}")
    init_image_tgt = Image.open(scout_image_path).convert("RGB")

    t_src = TF.to_tensor(init_image_src).unsqueeze(0).to(device=device, dtype=dtype) * 2.0 - 1.0
    t_tgt = TF.to_tensor(init_image_tgt).unsqueeze(0).to(device=device, dtype=dtype) * 2.0 - 1.0

    with torch.no_grad():
        z_src = pipe.vae.encode(t_src).latent_dist.sample() * pipe.vae.config.scaling_factor
        z_tgt = pipe.vae.encode(t_tgt).latent_dist.sample() * pipe.vae.config.scaling_factor

    patch_size = getattr(getattr(pipe, "transformer", None), "config", None)
    patch_size = getattr(patch_size, "patch_size", 2)
    H_lat, W_lat = z_src.shape[2], z_src.shape[3]
    H_p, W_p = H_lat // patch_size, W_lat // patch_size
    L_img = H_p * W_p

    src_cond, src_pooled, src_uncond, src_unpooled = encode_prompt_pair(pipe, source_prompt, device, max_t5_length=max_t5_length)
    tgt_cond, tgt_pooled, tgt_uncond, tgt_unpooled = encode_prompt_pair(pipe, target_prompt, device, max_t5_length=max_t5_length)

    use_unchanged_mode = len(unchanged_tokens) > 0
    if use_unchanged_mode:
        w_keep, _ = build_joint_text_weights(
            pipe=pipe,
            prompt=source_prompt,
            tokens_to_search=unchanged_tokens,
            w_clip=w_clip,
            w_t5=w_t5,
            max_t5_length=max_t5_length,
            token_match_mode=mask_extract_cfg["token_match_mode"],
            device=device,
        )
        if w_keep.sum() <= 0:
            raise ValueError(f"unchanged_tokens={unchanged_tokens} did not match any SD3 text tokens.")
    else:
        w_src, _ = build_joint_text_weights(
            pipe=pipe,
            prompt=source_prompt,
            tokens_to_search=source_mask_tokens,
            w_clip=w_clip,
            w_t5=w_t5,
            max_t5_length=max_t5_length,
            token_match_mode=mask_extract_cfg["token_match_mode"],
            device=device,
        )
        w_tgt, _ = build_joint_text_weights(
            pipe=pipe,
            prompt=target_prompt,
            tokens_to_search=target_mask_tokens,
            w_clip=w_clip,
            w_t5=w_t5,
            max_t5_length=max_t5_length,
            token_match_mode=mask_extract_cfg["token_match_mode"],
            device=device,
        )
        if len(source_mask_tokens) > 0 and w_src.sum() <= 0:
            raise ValueError(f"source_mask_tokens={source_mask_tokens} did not match any SD3 text tokens.")
        if len(target_mask_tokens) > 0 and w_tgt.sum() <= 0:
            raise ValueError(f"target_mask_tokens={target_mask_tokens} did not match any SD3 text tokens.")

    scheduler = pipe.scheduler
    scheduler.set_timesteps(T_steps, device=device)
    timesteps = scheduler.timesteps
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise RuntimeError("The current scheduler does not expose sigmas for the SD3 runner.")
    if len(sigmas) == len(timesteps) + 1:
        sigmas_steps = sigmas
    else:
        sigmas_steps = torch.cat([sigmas, sigmas[-1:]], dim=0)

    generator = torch.Generator(device=device).manual_seed(ref_noise_seed)
    scout_noise = randn_like_compat(z_src, generator=generator)

    zt_model = z_src.clone()
    zt_edit_full = z_src.clone()
    zt_visible = z_src.clone()

    W_accum_soft = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=torch.float32)
    W_accum_hard = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=torch.float32)
    C_accum = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=torch.float32)
    R_accum = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=torch.float32)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"sd3:{case['image_name']}:{case['target_code']}"):
        if T_steps - i > n_max:
            continue

        sigma_t = float(sigmas_steps[i])
        sigma_next = float(sigmas_steps[i + 1]) if (i + 1) < len(sigmas_steps) else sigma_t
        progress = i / T_steps

        if T_steps - i <= n_min:
            continue

        blur_sigma = blur_sigma_start - (blur_sigma_start - blur_sigma_end) * progress
        if use_unchanged_mode:
            M_norm, debug_maps = build_unchanged_complement_edit_map_sd3(
                pipe=pipe,
                x_src=z_src,
                scout_noise=scout_noise,
                src_prompt_embeds=src_cond,
                src_pooled=src_pooled,
                w_keep=w_keep,
                timestep=t,
                sigma_t=sigma_t,
                L_img=L_img,
                H_p=H_p,
                W_p=W_p,
                blur_sigma=blur_sigma,
                selected_layers=selected_layers,
                attn_reduce=mask_extract_cfg["attn_reduce"],
                norm_mode=mask_extract_cfg["norm_mode"],
            )
        else:
            M_norm, debug_maps = build_union_edit_map_sd3(
                pipe=pipe,
                x_src=z_src,
                x_tgt=z_tgt,
                scout_noise=scout_noise,
                src_prompt_embeds=src_cond,
                tgt_prompt_embeds=tgt_cond,
                src_pooled=src_pooled,
                tgt_pooled=tgt_pooled,
                w_src=w_src,
                w_tgt=w_tgt,
                timestep=t,
                sigma_t=sigma_t,
                L_img=L_img,
                H_p=H_p,
                W_p=W_p,
                blur_sigma=blur_sigma,
                selected_layers=selected_layers,
                attn_reduce=mask_extract_cfg["attn_reduce"],
                norm_mode=mask_extract_cfg["norm_mode"],
            )

        current_q = q_start + (q_end - q_start) * progress
        beta_t = torch.quantile(M_norm, current_q)
        beta_t = torch.clamp(beta_t, min=0.1, max=0.6)

        current_alpha = alpha_min + (alpha_max - alpha_min) * progress
        M_t = torch.sigmoid(current_alpha * (M_norm - beta_t))
        C_t = (M_t > tau_core).float()
        M_t = torch.where(C_t > 0, torch.tensor(1.0, dtype=M_t.dtype, device=M_t.device), M_t)

        r_t = int(ring_radius_start - (ring_radius_start - ring_radius_end) * progress)
        D_t = F.max_pool2d(C_t.unsqueeze(0).unsqueeze(0), kernel_size=2 * r_t + 1, stride=1, padding=r_t).squeeze()
        R_t = torch.clamp(D_t - C_t, 0.0, 1.0)
        W_raw = torch.clamp(M_t + w_ring * R_t * (1.0 - M_t), 0.0, 1.0)
        W_soft = slack + (1.0 - slack) * W_raw

        if save_maps:
            save_debug_maps(debug_maps, str(case_dir), i)
            Image.fromarray((W_raw.detach().cpu().numpy() * 255).astype("uint8")).save(case_dir / f"W_raw_step_{i:02d}.png")
            Image.fromarray((W_soft.detach().cpu().numpy() * 255).astype("uint8")).save(case_dir / f"W_soft_step_{i:02d}.png")

        W_raw_lat = F.interpolate(W_raw.unsqueeze(0).unsqueeze(0), size=(H_lat, W_lat), mode="bilinear", align_corners=False)
        W_soft_lat = F.interpolate(W_soft.unsqueeze(0).unsqueeze(0), size=(H_lat, W_lat), mode="bilinear", align_corners=False)
        C_lat = F.interpolate(C_t.unsqueeze(0).unsqueeze(0), size=(H_lat, W_lat), mode="nearest")
        R_lat = F.interpolate(R_t.unsqueeze(0).unsqueeze(0), size=(H_lat, W_lat), mode="bilinear", align_corners=False)

        V_delta_avg = torch.zeros_like(z_src, dtype=torch.float32)
        for _ in range(n_avg):
            fwd_noise = torch.randn_like(z_src)
            zt_src = (1 - sigma_t) * z_src + sigma_t * fwd_noise
            zt_tar = zt_model + zt_src - z_src

            Vt_src = calc_v_sd3(
                pipe=pipe,
                latents=zt_src,
                timestep=t,
                cond_embeds=src_cond,
                cond_pooled=src_pooled,
                uncond_embeds=src_uncond,
                uncond_pooled=src_unpooled,
                guidance_scale=src_guidance_scale,
            )
            Vt_tar = calc_v_sd3(
                pipe=pipe,
                latents=zt_tar,
                timestep=t,
                cond_embeds=tgt_cond,
                cond_pooled=tgt_pooled,
                uncond_embeds=tgt_uncond,
                uncond_pooled=tgt_unpooled,
                guidance_scale=tgt_guidance_scale,
            )
            V_delta_avg += (1.0 / n_avg) * (Vt_tar - Vt_src)

        zt_edit_full = zt_edit_full.to(torch.float32)
        zt_edit_full = zt_edit_full + (sigma_next - sigma_t) * V_delta_avg

        source_anchor = z_src.to(torch.float32)
        model_anchor = source_anchor
        visible_anchor = source_anchor

        W_accum_soft = torch.max(W_accum_soft, W_soft_lat.to(W_accum_soft.dtype))
        W_accum_hard = torch.max(W_accum_hard, W_raw_lat.to(W_accum_hard.dtype))
        C_accum = torch.max(C_accum, C_lat.to(C_accum.dtype))
        R_accum = torch.max(R_accum, R_lat.to(R_accum.dtype))
        W_soft_state = W_accum_soft
        W_hard_state = W_accum_hard
        C_state = C_accum
        R_state = R_accum

        zt_model = model_anchor + anchor_rho * W_soft_state * (zt_edit_full - model_anchor)

        core_keep = C_state
        ring_only = torch.clamp(R_state * (1.0 - core_keep), 0.0, 1.0)
        outer_only = torch.clamp(1.0 - core_keep - ring_only, 0.0, 1.0)

        z_static_visible = model_anchor + anchor_rho * W_soft_state * (zt_edit_full - model_anchor)
        z_ring_visible = visible_anchor + anchor_rho * (ring_blend * W_hard_state) * (zt_edit_full - visible_anchor)
        zt_visible = core_keep * z_static_visible + ring_only * z_ring_visible + outer_only * visible_anchor

        zt_model = zt_model.to(dtype)
        zt_edit_full = zt_edit_full.to(dtype)
        zt_visible = zt_visible.to(dtype)

    with torch.no_grad():
        img = pipe.vae.decode(zt_visible / pipe.vae.config.scaling_factor, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)

    out_img = TF.to_pil_image(img[0].detach().cpu())
    out_path = case_output_path(case_dir, case["target_code"])
    out_img.save(out_path)

    save_case_manifest(
        case_dir,
        case,
        extra={
            "mode": "sd3",
            "scout_image_path": str(scout_image_path),
            "sam_flow": sam_cfg,
        },
    )
    return out_path


def _collect_cases(config: Dict, image_names: List[str] | None, target_codes: List[str] | None) -> List[Dict]:
    dataset_path = config["dataset"]["path"]
    data_root = config["dataset"]["root"]
    strict = bool(config["dataset"].get("strict_validation", False))
    records, warnings = load_dataset(dataset_path, strict=strict)
    for warning in warnings:
        print(f"[dataset warning] {warning}")

    selected_records = filter_records(
        records,
        image_names=image_names or config.get("run", {}).get("image_names"),
        start_index=config.get("run", {}).get("start_index"),
        end_index=config.get("run", {}).get("end_index"),
        max_records=config.get("run", {}).get("max_records"),
    )
    return iter_cases(
        selected_records,
        data_root=data_root,
        target_codes=target_codes or config.get("run", {}).get("target_codes"),
        max_cases=config.get("run", {}).get("max_cases"),
    )


def main():
    parser = argparse.ArgumentParser(description="Run SAM-Flow with the SD3 backbone.")
    parser.add_argument("--config", default="configs/sd3.yaml")
    parser.add_argument("--image-name", action="append", dest="image_names")
    parser.add_argument("--target-code", action="append", dest="target_codes")
    args = parser.parse_args()

    config = load_yaml(args.config)
    cases = _collect_cases(config, args.image_names, args.target_codes)
    if not cases:
        raise SystemExit("No SD3 cases matched the current filters.")

    pipe = load_sd3_pipeline(config)
    results_root = config["results"]["root"]
    overwrite = bool(config.get("run", {}).get("overwrite", False))

    for case in cases:
        image_dir, case_dir = prepare_case_dirs(results_root, case["image_name"], case["target_code"])
        out_path = case_output_path(case_dir, case["target_code"])
        if out_path.exists() and not overwrite:
            print(f"[sam_flow_sd3] skip existing {out_path}")
            continue

        scout_path = case_scout_path(case_dir, case["target_code"])
        if overwrite or not scout_path.exists():
            scout_path = generate_scout_image_sd3(pipe, case, config, image_dir, case_dir)

        out_path = run_sam_flow_sd3_case(pipe, case, config, scout_path, case_dir)
        print(f"[sam_flow_sd3] saved {case['image_name']} / {case['target_code']} -> {out_path}")


if __name__ == "__main__":
    main()
