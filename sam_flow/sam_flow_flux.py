
import argparse
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from PIL import Image
from tqdm import tqdm

from .flowedit_flux import generate_scout_image_flux, load_flux_pipeline
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

# =====================================================================

# =====================================================================
attn_store = []
original_sdpa = F.scaled_dot_product_attention


def get_hooked_sdpa(L_img):
    def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        out = original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )
        L_seq = query.shape[-2]
        if L_seq > L_img:
            L_txt = L_seq - L_img
            with torch.no_grad():
                q_img = query[:, :, L_txt:, :].float()
                k_all = key.float()
                scale_factor = scale or (1.0 / query.shape[-1] ** 0.5)
                attn_weight_row = torch.matmul(q_img, k_all.transpose(-2, -1)) * scale_factor
                if attn_mask is not None:
                    attn_weight_row += attn_mask[:, :, L_txt:, :].float()
                attn_prob_row = torch.softmax(attn_weight_row, dim=-1)
                A_IT = attn_prob_row[:, :, :, :L_txt].detach().cpu()
                A = A_IT.mean(dim=1)
                attn_store.append(A)
        return out

    return hooked_sdpa


def get_token_indices(pipe, prompt, tokens_to_search, max_len=512):
    """
    Robust token index finder for FLUX text tokens.
    - Uses tokenizer input_ids (NOT txt_ids) and matches by sub-sequence of token ids.
    - Handles cases like 'tiger' being split into multiple sub-tokens.
    - Preference order: tokenizer_2 -> tokenizer. Returns the first non-empty match.
    """
    if not tokens_to_search:
        return []

    def _find_subseq(haystack, needle):
        if not needle:
            return []
        out = []
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            if haystack[i:i+n] == needle:
                out.extend(range(i, i+n))
        return out

    tokenizer_names = []
    if getattr(pipe, 'tokenizer_2', None) is not None:
        tokenizer_names.append('tokenizer_2')
    if getattr(pipe, 'tokenizer', None) is not None:
        tokenizer_names.append('tokenizer')
    if not tokenizer_names:
        raise RuntimeError('No tokenizer found on pipe (expected tokenizer_2 or tokenizer).')

    for name in tokenizer_names:
        tok = getattr(pipe, name)
        enc = tok(
            prompt,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        full_ids = enc.input_ids[0].tolist()
        special = set(getattr(tok, 'all_special_ids', []))

        idx = []
        for t_search in tokens_to_search:
            for w in str(t_search).replace('_', ' ').split():
                needle = tok(w, add_special_tokens=False).input_ids
                idx.extend(_find_subseq(full_ids, needle))

        # remove special tokens
        idx = sorted({i for i in idx if full_ids[i] not in special})
        if len(idx) > 0:
            return idx

    return []

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    timestep = t.expand(latents.shape[0])
    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
    return noise_pred


# =====================================================================

def min_max_norm(t_map):
    v_min, v_max = t_map.min(), t_map.max()
    if (v_max - v_min) > 1e-6:
        return (t_map - v_min) / (v_max - v_min)
    return torch.zeros_like(t_map)

def run_attention_scout(
    pipe,
    scout_latents,
    scout_embeds,
    scout_pooled,
    scout_txt_ids,
    scout_img_ids,
    scout_guides,
    t,
    L_img,
    selected_layers,
):
    attn_store.clear()
    F.scaled_dot_product_attention = get_hooked_sdpa(L_img=L_img)
    # diffusers FLUX expects `txt_ids` / `img_ids` to be 2D tensors of shape (seq_len, 3).
    # Some forks pass a batch dimension (B, seq_len, 3). Only strip the *batch* dimension when it exists.
    if scout_txt_ids is not None and scout_txt_ids.ndim == 3:
        scout_txt_ids = scout_txt_ids[0]
    if scout_img_ids is not None and scout_img_ids.ndim == 3:
        scout_img_ids = scout_img_ids[0]

    try:
        with torch.no_grad():
            _ = pipe.transformer(
                hidden_states=scout_latents,
                timestep=(t / 1000).expand(scout_latents.shape[0]),
                guidance=scout_guides,
                encoder_hidden_states=scout_embeds,
                txt_ids=scout_txt_ids,
                img_ids=scout_img_ids,
                pooled_projections=scout_pooled,
                return_dict=False,
            )[0]
    finally:
        F.scaled_dot_product_attention = original_sdpa

    if len(attn_store) == 0:
        raise RuntimeError("Attention hook did not capture any FLUX attention outputs.")

    valid_layers = [j for j in selected_layers if 0 <= j < len(attn_store)]
    if len(valid_layers) == 0:
        raise RuntimeError(
            f"Only {len(attn_store)} attention layers were captured, which is insufficient for {selected_layers}."
        )

    A_t_layers = torch.stack([attn_store[j] for j in valid_layers])
    A_t = A_t_layers.mean(dim=0)
    attn_store.clear()
    return A_t


def extract_token_map(attn_map, token_indices, H_p, W_p, current_sigma, device):
    if len(token_indices) == 0:
        return torch.zeros((H_p, W_p), device=device)

    M = attn_map[:, token_indices].sum(dim=-1).view(H_p, W_p).float()
    M = TF.gaussian_blur(M.unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=current_sigma).squeeze()
    return min_max_norm(M).to(device)


def build_union_edit_map(
    pipe,
    x_src_packed,
    x_tgt_packed,
    scout_noise,
    src_prompt_embeds,
    tgt_prompt_embeds,
    src_pooled_embeds,
    tgt_pooled_embeds,
    src_text_ids,
    tgt_text_ids,
    img_ids,
    src_guidance,
    tgt_guidance,
    src_indices,
    tgt_indices,
    t,
    t_i,
    L_img,
    H_p,
    W_p,
    current_sigma,
):
    noisy_src = (1 - t_i) * x_src_packed + t_i * scout_noise
    noisy_tgt = (1 - t_i) * x_tgt_packed + t_i * scout_noise

    scout_latents = torch.cat([noisy_src, noisy_tgt], dim=0)
    scout_embeds = torch.cat([src_prompt_embeds, tgt_prompt_embeds], dim=0)
    scout_pooled = torch.cat([src_pooled_embeds, tgt_pooled_embeds], dim=0)
    scout_guides = torch.cat([tgt_guidance, tgt_guidance], dim=0) if src_guidance is not None else None

    A_t = run_attention_scout(
        pipe=pipe,
        scout_latents=scout_latents,
        scout_embeds=scout_embeds,
        scout_pooled=scout_pooled,
        scout_txt_ids=src_text_ids,
        scout_img_ids=img_ids,
        scout_guides=scout_guides,
        t=t,
        L_img=L_img,
        selected_layers=list(range(4, 19)),
    )
    A_src, A_tgt = A_t[0], A_t[1]

    M_src_norm = extract_token_map(A_src, src_indices, H_p, W_p, current_sigma, x_src_packed.device)
    M_tgt_norm = extract_token_map(A_tgt, tgt_indices, H_p, W_p, current_sigma, x_src_packed.device)
    M_norm = torch.max(M_src_norm, M_tgt_norm)

    debug_maps = {
        "mode": "union",
        "src_token_map": M_src_norm,
        "tgt_token_map": M_tgt_norm,
        "base_map": M_norm,
    }
    return M_norm, debug_maps


def build_unchanged_complement_edit_map(
    pipe,
    x_src_packed,
    scout_noise,
    src_prompt_embeds,
    src_pooled_embeds,
    src_text_ids,
    tgt_text_ids,
    img_ids,
    tgt_guidance,
    unchanged_indices,
    t,
    t_i,
    L_img,
    H_p,
    W_p,
    current_sigma,
):
    noisy_src = (1 - t_i) * x_src_packed + t_i * scout_noise

    A_t = run_attention_scout(
        pipe=pipe,
        scout_latents=noisy_src,
        scout_embeds=src_prompt_embeds,
        scout_pooled=src_pooled_embeds,
        scout_txt_ids=src_text_ids,
        scout_img_ids=img_ids,
        scout_guides=tgt_guidance,
        t=t,
        L_img=L_img,
        selected_layers=list(range(4, 19)),
    )
    A_src = A_t[0]

    M_keep_norm = extract_token_map(A_src, unchanged_indices, H_p, W_p, current_sigma, x_src_packed.device)
    M_norm = 1.0 - M_keep_norm
    M_norm = torch.clamp(M_norm, min=0.0, max=1.0)

    debug_maps = {
        "mode": "unchanged_complement",
        "keep_token_map": M_keep_norm,
        "base_map": M_norm,
    }
    return M_norm, debug_maps


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


# =====================================================================
# =====================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
def run_sam_flow_flux_case(
    pipe: FluxPipeline,
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

    source_prompt = case["source_prompt"]
    target_prompt = case["target_prompt"]
    source_mask_tokens = case["source_mask_tokens"]
    target_mask_tokens = case["target_mask_tokens"]
    unchanged_tokens = case["unchanged_tokens"]

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
    ring_blend = float(sam_cfg.get("ring_blend", 0.35))

    init_image_src = Image.open(case["image_path"]).convert("RGB")
    if not Path(scout_image_path).exists():
        raise FileNotFoundError(f"Scout image not found: {scout_image_path}")
    init_image_tgt = Image.open(scout_image_path).convert("RGB")

    t_src = TF.to_tensor(init_image_src).unsqueeze(0).to(device=device, dtype=dtype) * 2.0 - 1.0
    t_tgt = TF.to_tensor(init_image_tgt).unsqueeze(0).to(device=device, dtype=dtype) * 2.0 - 1.0

    with torch.no_grad():
        z_src = pipe.vae.encode(t_src).latent_dist.sample()
        z_src = (z_src - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        z_tgt = pipe.vae.encode(t_tgt).latent_dist.sample()
        z_tgt = (z_tgt - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    num_channels_latents = pipe.transformer.config.in_channels // 4
    x_src_packed = pipe._pack_latents(z_src, 1, num_channels_latents, z_src.shape[2], z_src.shape[3])
    x_tgt_packed = pipe._pack_latents(z_tgt, 1, num_channels_latents, z_tgt.shape[2], z_tgt.shape[3])
    if x_src_packed.shape != x_tgt_packed.shape:
        raise RuntimeError(
            f"Packed latent shape mismatch: source={tuple(x_src_packed.shape)} target={tuple(x_tgt_packed.shape)}."
        )

    H_p = z_src.shape[2] // 2
    W_p = z_src.shape[3] // 2
    L_img = int(x_src_packed.shape[1])
    if L_img != H_p * W_p:
        ratio = float(z_src.shape[2]) / float(z_src.shape[3] + 1e-8)
        best = None
        for h in range(1, int(math.sqrt(L_img)) + 1):
            if L_img % h != 0:
                continue
            w = L_img // h
            r = h / float(w)
            score = abs(math.log((r + 1e-8) / (ratio + 1e-8)))
            if best is None or score < best[0]:
                best = (score, h, w)
        if best is None:
            raise RuntimeError(f"Cannot factorize L_img={L_img} into a 2D grid.")
        _, H_p, W_p = best

    img_ids = pipe._prepare_latent_image_ids(1, H_p, W_p, device, x_src_packed.dtype)
    if img_ids.shape[0] != L_img:
        raise RuntimeError(f"img_ids length mismatch: img_ids={tuple(img_ids.shape)} vs L_img={L_img}")

    src_prompt_embeds, src_pooled_embeds, src_text_ids = pipe.encode_prompt(
        prompt=source_prompt,
        prompt_2=None,
        device=device,
        max_sequence_length=512,
    )
    tgt_prompt_embeds, tgt_pooled_embeds, tgt_text_ids = pipe.encode_prompt(
        prompt=target_prompt,
        prompt_2=None,
        device=device,
        max_sequence_length=512,
    )

    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(1)
        tgt_guidance = torch.tensor([tgt_guidance_scale], device=device).expand(1)
    else:
        src_guidance = None
        tgt_guidance = None

    use_unchanged_mode = len(unchanged_tokens) > 0
    src_indices = get_token_indices(pipe, source_prompt, source_mask_tokens)
    tgt_indices = get_token_indices(pipe, target_prompt, target_mask_tokens)
    unchanged_indices = get_token_indices(pipe, source_prompt, unchanged_tokens)
    if len(src_indices) == 0 and len(source_mask_tokens) > 0:
        print(f"[sam_flow_flux warning] no source token match for {source_mask_tokens}")
    if len(tgt_indices) == 0 and len(target_mask_tokens) > 0:
        print(f"[sam_flow_flux warning] no target token match for {target_mask_tokens}")
    if use_unchanged_mode and len(unchanged_indices) == 0:
        raise ValueError(f"unchanged_tokens={unchanged_tokens} did not match any source tokens.")

    scheduler = pipe.scheduler
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    mu = calculate_shift(
        x_src_packed.shape[1],
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(scheduler, T_steps, device, sigmas=sigmas, mu=mu)

    generator = torch.Generator(device=device).manual_seed(ref_noise_seed)
    scout_noise = torch.randn(x_src_packed.shape, generator=generator, device=device, dtype=x_src_packed.dtype)

    zt_model = x_src_packed.clone()
    zt_edit_full = x_src_packed.clone()
    zt_visible = x_src_packed.clone()

    W_accum_soft = torch.zeros((1, L_img, 1), device=device, dtype=zt_model.dtype)
    W_accum_hard = torch.zeros((1, L_img, 1), device=device, dtype=zt_model.dtype)
    C_accum = torch.zeros((1, L_img, 1), device=device, dtype=zt_model.dtype)
    R_accum = torch.zeros((1, L_img, 1), device=device, dtype=zt_model.dtype)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"flux:{case['image_name']}:{case['target_code']}"):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i
        progress = i / T_steps

        if T_steps - i <= n_min:
            continue

        current_sigma = blur_sigma_start - (blur_sigma_start - blur_sigma_end) * progress
        if use_unchanged_mode:
            M_norm, debug_maps = build_unchanged_complement_edit_map(
                pipe=pipe,
                x_src_packed=x_src_packed,
                scout_noise=scout_noise,
                src_prompt_embeds=src_prompt_embeds,
                src_pooled_embeds=src_pooled_embeds,
                src_text_ids=src_text_ids,
                tgt_text_ids=tgt_text_ids,
                img_ids=img_ids,
                tgt_guidance=tgt_guidance,
                unchanged_indices=unchanged_indices,
                t=t,
                t_i=t_i,
                L_img=L_img,
                H_p=H_p,
                W_p=W_p,
                current_sigma=current_sigma,
            )
        else:
            M_norm, debug_maps = build_union_edit_map(
                pipe=pipe,
                x_src_packed=x_src_packed,
                x_tgt_packed=x_tgt_packed,
                scout_noise=scout_noise,
                src_prompt_embeds=src_prompt_embeds,
                tgt_prompt_embeds=tgt_prompt_embeds,
                src_pooled_embeds=src_pooled_embeds,
                tgt_pooled_embeds=tgt_pooled_embeds,
                src_text_ids=src_text_ids,
                tgt_text_ids=tgt_text_ids,
                img_ids=img_ids,
                src_guidance=src_guidance,
                tgt_guidance=tgt_guidance,
                src_indices=src_indices,
                tgt_indices=tgt_indices,
                t=t,
                t_i=t_i,
                L_img=L_img,
                H_p=H_p,
                W_p=W_p,
                current_sigma=current_sigma,
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
        R_t = torch.clamp(D_t - C_t, min=0.0, max=1.0)
        W_raw = torch.clamp(M_t + w_ring * R_t * (1.0 - M_t), min=0.0, max=1.0)
        W_soft = slack + (1.0 - slack) * W_raw

        W_raw_flat = W_raw.view(1, L_img, 1).to(device=device, dtype=zt_model.dtype)
        W_soft_flat = W_soft.view(1, L_img, 1).to(device=device, dtype=zt_model.dtype)
        C_t_flat = C_t.view(1, L_img, 1).to(device=device, dtype=zt_model.dtype)
        R_t_flat = R_t.view(1, L_img, 1).to(device=device, dtype=zt_model.dtype)

        if save_maps:
            save_debug_maps(debug_maps, str(case_dir), i)
            Image.fromarray((W_raw.detach().cpu().numpy() * 255).astype("uint8")).save(case_dir / f"W_raw_step_{i:02d}.png")
            Image.fromarray((W_soft.detach().cpu().numpy() * 255).astype("uint8")).save(case_dir / f"W_soft_step_{i:02d}.png")

        V_delta_avg = torch.zeros_like(x_src_packed)
        for _ in range(n_avg):
            fwd_noise = torch.randn_like(x_src_packed)
            zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise
            zt_tar = zt_model + zt_src - x_src_packed

            Vt_src = calc_v_flux(
                pipe,
                zt_src,
                src_prompt_embeds,
                src_pooled_embeds,
                src_guidance,
                src_text_ids,
                img_ids,
                t,
            )
            Vt_tar = calc_v_flux(
                pipe,
                zt_tar,
                tgt_prompt_embeds,
                tgt_pooled_embeds,
                tgt_guidance,
                tgt_text_ids,
                img_ids,
                t,
            )
            V_delta_avg += (1.0 / n_avg) * (Vt_tar - Vt_src)

        zt_edit_full = zt_edit_full.to(torch.float32)
        zt_edit_full = zt_edit_full + (t_im1 - t_i) * V_delta_avg

        source_anchor = x_src_packed.to(torch.float32)
        model_anchor = source_anchor
        visible_anchor = source_anchor

        W_accum_soft = torch.max(W_accum_soft, W_soft_flat)
        W_accum_hard = torch.max(W_accum_hard, W_raw_flat)
        C_accum = torch.max(C_accum, C_t_flat)
        R_accum = torch.max(R_accum, R_t_flat)
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

        zt_edit_full = zt_edit_full.to(V_delta_avg.dtype)
        zt_model = zt_model.to(V_delta_avg.dtype)
        zt_visible = zt_visible.to(V_delta_avg.dtype)

    unpacked_out = pipe._unpack_latents(
        zt_visible,
        init_image_src.height,
        init_image_src.width,
        pipe.vae_scale_factor,
    )
    with torch.no_grad():
        image = pipe.vae.decode(
            unpacked_out / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor,
            return_dict=False,
        )[0]

    edited_image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    out_path = case_output_path(case_dir, case["target_code"])
    edited_image.save(out_path)

    save_case_manifest(
        case_dir,
        case,
        extra={
            "mode": "flux",
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
    parser = argparse.ArgumentParser(description="Run SAM-Flow with the FLUX backbone.")
    parser.add_argument("--config", default="configs/flux.yaml")
    parser.add_argument("--image-name", action="append", dest="image_names")
    parser.add_argument("--target-code", action="append", dest="target_codes")
    args = parser.parse_args()

    config = load_yaml(args.config)
    cases = _collect_cases(config, args.image_names, args.target_codes)
    if not cases:
        raise SystemExit("No FLUX cases matched the current filters.")

    pipe = load_flux_pipeline(config)
    results_root = config["results"]["root"]
    overwrite = bool(config.get("run", {}).get("overwrite", False))

    for case in cases:
        image_dir, case_dir = prepare_case_dirs(results_root, case["image_name"], case["target_code"])
        out_path = case_output_path(case_dir, case["target_code"])
        if out_path.exists() and not overwrite:
            print(f"[sam_flow_flux] skip existing {out_path}")
            continue

        scout_path = case_scout_path(case_dir, case["target_code"])
        if overwrite or not scout_path.exists():
            scout_path = generate_scout_image_flux(pipe, case, config, image_dir, case_dir)

        out_path = run_sam_flow_flux_case(pipe, case, config, scout_path, case_dir)
        print(f"[sam_flow_flux] saved {case['image_name']} / {case['target_code']} -> {out_path}")


if __name__ == "__main__":
    main()
