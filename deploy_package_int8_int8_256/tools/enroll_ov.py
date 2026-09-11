#!/usr/bin/env python3
"""OV-only voice enrollment (torch 불필요):
voice wav -> mimi 인코더 OV -> step OV로 voice prefill -> assets/voice_cache.npz 갱신.
필요 패키지: openvino, numpy, scipy

사용:
    python tools/enroll_ov.py --voice /path/to/new_voice.wav --pkg .
"""
import argparse
import json
import math
import os
import sys
import wave

import numpy as np

assert "torch" not in sys.modules, "torch 없이 동작해야 합니다"

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 24000
L = 240000  # 10s
FRAME = 1920


def load_voice_24k(path):
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(-1)
    d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        d = resample_poly(d, SR // g, sr // g).astype(np.float32)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--pkg", default=PKG_DIR)
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()
    import openvino as ov

    M = os.path.join(a.pkg, "models")
    A = os.path.join(a.pkg, "assets")
    T = os.path.join(a.pkg, "tools")
    core = ov.Core()
    enc = core.compile_model(os.path.join(T, "mimi_enc.xml"), "CPU")
    step = core.compile_model(os.path.join(M, "step.xml"), "CPU")
    P = json.load(open(os.path.join(A, "params.json")))

    wav = load_voice_24k(a.voice)
    if len(wav) > L:
        print(f"10초로 자름 ({len(wav) / SR:.1f}s -> 10s)")
        wav = wav[:L]
    n = len(wav)
    pad_frame = (FRAME - n % FRAME) % FRAME
    N = (n + pad_frame) // FRAME  # voice latent frames
    print(f"samples={n}, voice frames={N}")
    assert 1 + N <= a.max_len, "voice가 너무 김"

    inp = np.zeros((1, 1, L), dtype=np.float32)
    inp[0, 0, :n] = wav  # 나머지는 0 (뒤쪽 패딩은 인과성으로 결과에 영향 없음)
    # 프레임 경계 패딩 복제 (torch pad_for_conv1d과 동일)
    if pad_frame:
        inp[0, 0, n:n + pad_frame] = 0.0
    cond = np.asarray(enc({"wav": inp})[enc.output("cond")])[:, :N, :]
    print(f"conditioning: {cond.shape}")

    bos = np.load(os.path.join(T, "bos_voice.npy"))  # [1,1,1024]
    prompt = np.concatenate([bos, cond.astype(np.float32)], axis=1)
    Tp = prompt.shape[1]

    # 무상태에서 voice prefill
    offs, caches = [], []
    for i in range(P["n_layers"]):
        offs.append(np.zeros(1, dtype=np.int64))
        shp = tuple(step.input(f"cache{i}").shape)
        caches.append(np.full(shp, np.nan, dtype=np.float32))
    nan_tok = np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32)
    is_text = np.ones(1, dtype=bool)
    for t in range(Tp):
        feed = {"token": nan_tok, "emb": prompt[:, t:t + 1, :], "is_text": is_text}
        for i in range(P["n_layers"]):
            feed[f"off{i}"] = offs[i]
            feed[f"cache{i}"] = caches[i]
        res = step(feed)
        for i in range(P["n_layers"]):
            caches[i] = np.asarray(res[step.output(f"new_cache{i}")])
            offs[i] = offs[i] + 1
    print(f"voice offset={int(offs[0][0])}")

    out = {"off0": np.array([int(offs[0][0])], dtype=np.int64)}
    for i in range(P["n_layers"]):
        out[f"cache{i}"] = caches[i]
    np.savez(os.path.join(A, "voice_cache.npz"), **out)
    print(f"저장: assets/voice_cache.npz ({sum(v.nbytes for v in out.values()) / 1e6:.1f}MB)")
    P["voice"] = a.voice
    json.dump(P, open(os.path.join(A, "params.json"), "w"), indent=2)
    print("params.json voice 갱신")


if __name__ == "__main__":
    main()
