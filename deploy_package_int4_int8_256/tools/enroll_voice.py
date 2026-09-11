#!/usr/bin/env python3
"""Voice enrollment (빌드타임 전용, torch + 원본 모델 필요):
새 voice wav -> voice prefill -> assets/voice_cache.npz 갱신.
cond_embed.npy / tokenizer.model / params.json도 함께 갱신한다.

사용:
    python tools/enroll_voice.py --voice /path/to/new_voice.wav --pkg .
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

CONFIG = "hf://seastar105/pocket-tts-korean-300m/korean.yaml"
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="새 voice wav 경로")
    ap.add_argument("--pkg", default=PKG_DIR, help="deploy_package 경로")
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()
    A = os.path.join(a.pkg, "assets")
    os.makedirs(A, exist_ok=True)
    from pocket_tts import TTSModel

    torch.set_num_threads(8)
    print("모델 로드...")
    model = TTSModel.load_model(config=CONFIG)
    model.eval()
    flow = model.flow_lm

    vs = model.get_state_for_audio_prompt(a.voice)
    names = [ly.self_attn._module_absolute_name for ly in flow.transformer.layers]
    off0 = int(vs[names[0]]["offset"].view(-1)[0].item())
    T = vs[names[0]]["cache"].shape[2]
    print(f"voice offset={off0}, cache T={T}")
    assert T <= a.max_len, f"voice 캐시 {T} > MAX {a.max_len}"

    out = {"off0": np.array([off0], dtype=np.int64)}
    for i, n in enumerate(names):
        s = vs[n]
        c = s["cache"][:, :1].detach().numpy().astype(np.float32)
        pad = np.full((2, 1, a.max_len - c.shape[2]) + c.shape[3:], np.nan, dtype=np.float32)
        out[f"cache{i}"] = np.concatenate([c, pad], axis=2)
    np.savez(os.path.join(A, "voice_cache.npz"), **out)
    print(f"저장: assets/voice_cache.npz ({sum(v.nbytes for v in out.values()) / 1e6:.1f}MB)")

    emb = flow.conditioner.embed.weight.detach().numpy().astype(np.float32)
    np.save(os.path.join(A, "cond_embed.npy"), emb)
    print(f"저장: assets/cond_embed.npy ({emb.nbytes / 1e6:.1f}MB, shape={emb.shape})")

    import glob
    cands = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--seastar105--pocket-tts-korean-300m/snapshots/*/tokenizer.model"))
    assert cands, "tokenizer.model을 HF 캐시에서 찾지 못함"
    shutil.copy(cands[0], os.path.join(A, "tokenizer.model"))
    print(f"저장: assets/tokenizer.model ({os.path.getsize(cands[0])}B)")

    params = dict(
        temp=model.temp, eos_threshold=model.eos_threshold,
        sample_rate=model.sample_rate, ldim=flow.ldim, dim=flow.dim,
        emb_std=flow.emb_std.detach().numpy().astype(np.float32).tolist(),
        emb_mean=flow.emb_mean.detach().numpy().astype(np.float32).tolist(),
        max_len=a.max_len, max_mimi_t=2048, steps_per=16,
        tokens_per_sec=3.0, gen_pad_sec=2.0, frame_rate=12.5,
        pad_spaces=model.pad_with_spaces_for_short_inputs,
        rm_semi=model.remove_semicolons, append_punct=model.append_terminal_punctuation,
        n_layers=len(names), voice=a.voice,
    )
    with open(os.path.join(A, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print("저장: assets/params.json")


if __name__ == "__main__":
    main()
