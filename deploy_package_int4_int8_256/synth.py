#!/usr/bin/env python3
"""pocket-tts 한국어 TTS — 전체 OpenVINO 합성 (torch 불필요).

필요 패키지: openvino, numpy, scipy, sentencepiece

사용:
    python synth.py --text "안녕하세요." --out out.wav
    python synth.py --text "안녕하세요." --out out.wav --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.io.wavfile

# torch 의존성 차단 (패키지 자립 보장)
import sys
assert "torch" not in sys.modules, "torch 없이 동작해야 합니다"

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_text_prompt(text, pad_spaces, rm_semi, append_punct=True):
    text = text.strip()
    if text == "":
        raise ValueError("empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if rm_semi:
        text = text.replace(";", ",")
    nw = len(text.split())
    frames_after = 3 if nw <= 4 else 1
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if append_punct and text[-1].isalnum():
        text = text + "."
    if pad_spaces and len(text.split()) < 5:
        text = " " * 8 + text
    return text, frames_after


def main():
    ap = argparse.ArgumentParser(description="pocket-tts OV 합성")
    ap.add_argument("--pkg", default=PKG_DIR, help="deploy_package 경로")
    ap.add_argument("--text", default="안녕하세요. 한국어 음성 합성 모델입니다.")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import sentencepiece as spm
    import openvino as ov

    M = os.path.join(a.pkg, "models")
    A = os.path.join(a.pkg, "assets")
    P = json.load(open(os.path.join(A, "params.json")))
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    core = ov.Core()
    step = core.compile_model(os.path.join(M, "step.xml"), "CPU")
    sampler = core.compile_model(os.path.join(M, "sampler.xml"), "CPU")
    mimi = core.compile_model(os.path.join(M, "mimi.xml"), "CPU")
    print(f"OV 로드: {time.perf_counter() - t0:.1f}s")

    vc = np.load(os.path.join(A, "voice_cache.npz"))
    offs = [vc["off0"].copy() for _ in range(P["n_layers"])]
    caches = [vc[f"cache{i}"] for i in range(P["n_layers"])]
    off0 = int(vc["off0"][0])
    MAX = P["max_len"]

    # --- 텍스트 조건 ---
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(A, "tokenizer.model"))
    fmt, guess = prepare_text_prompt(a.text, P["pad_spaces"], P["rm_semi"], P["append_punct"])
    ids = sp.encode(fmt, out_type=int)
    tc = len(ids)
    embed = np.load(os.path.join(A, "cond_embed.npy"))
    text_emb = embed[np.array(ids, dtype=np.int64)][None, :, :].astype(np.float32)
    max_gen = int(np.ceil((tc / P["tokens_per_sec"] + P["gen_pad_sec"]) * P["frame_rate"]))
    print(f"text tokens={tc}, max_gen={max_gen}, voice off={off0}")
    assert off0 + tc + max_gen <= MAX, "MAX_LEN 초과: 더 긴 컨텍스트 모델 필요"

    F_ = np.zeros(1, dtype=bool)
    T_ = np.ones(1, dtype=bool)
    nan_tok = np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32)
    zero_emb = np.zeros((1, 1, P["dim"]), dtype=np.float32)

    def run_step(tok, emb, flag):
        feed = {"token": tok, "emb": emb, "is_text": flag}
        for i in range(P["n_layers"]):
            feed[f"off{i}"] = offs[i]
            feed[f"cache{i}"] = caches[i]
        return step(feed)

    # --- text prefill ---
    for t in range(tc):
        res = run_step(nan_tok, text_emb[:, t:t + 1, :], T_)
        for i in range(P["n_layers"]):
            caches[i] = np.asarray(res[step.output(f"new_cache{i}")])
            offs[i] = offs[i] + 1

    # --- AR 루프 ---
    std = P["temp"] ** 0.5
    token = np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32)
    latents = []
    eos_step = None
    frames_after = guess + 2
    t1 = time.perf_counter()
    for s in range(max_gen):
        res = run_step(token, zero_emb, F_)
        tout = np.asarray(res[step.output("out")])  # [1,1,dim]
        for i in range(P["n_layers"]):
            caches[i] = np.asarray(res[step.output(f"new_cache{i}")])
            offs[i] = offs[i] + 1
        last = tout[:, -1:]
        noise = rng.normal(0, std, size=(1, P["ldim"])).astype(np.float32)
        sr = sampler({"cond": last.reshape(1, -1), "noise": noise})
        lat = np.asarray(sr[sampler.output("latent")])
        eos = float(np.asarray(sr[sampler.output("eos")])[0, 0])
        if eos > P["eos_threshold"] and eos_step is None:
            eos_step = s
        if eos_step is not None and s >= eos_step + frames_after:
            break
        latents.append(lat)
        token = lat[:, None, :]
    gen_t = time.perf_counter() - t1
    print(f"AR: {len(latents)} frames, {gen_t:.2f}s ({gen_t / max(1, len(latents)) * 1000:.1f}ms/step)")

    # --- Mimi 디코딩 ---
    mi = mimi.inputs
    mstate = {}
    for inp in mi[1:]:
        n, shp = inp.any_name, tuple(inp.shape)
        if "off" in n:
            mstate[n] = np.zeros(shp, dtype=np.int64)
        elif "cache" in n:
            mstate[n] = np.full(shp, np.nan, dtype=np.float32)
        else:
            mstate[n] = np.zeros(shp, dtype=np.float32)
    in_names = [inp.any_name for inp in mi[1:]]  # 출력 n{i} <-> 입력 in_names[i]
    estd = np.array(P["emb_std"], dtype=np.float32)
    emean = np.array(P["emb_mean"], dtype=np.float32)
    chunks = []
    for lat in latents:
        f = dict(mstate)
        f["latent"] = (lat.reshape(1, 1, -1) * estd + emean).astype(np.float32)
        res = mimi(f)
        chunks.append(np.asarray(res[mimi.output("audio")])[0, 0])
        for j, nm in enumerate(in_names):
            mstate[nm] = np.asarray(res[mimi.output(f"n{j}")])
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    dur = len(audio) / P["sample_rate"]
    scipy.io.wavfile.write(a.out, P["sample_rate"], audio)
    print(f"저장: {a.out} ({dur:.2f}s 오디오, {dur / gen_t:.2f}x)")
    print(f"peak={np.abs(audio).max():.3f}")


if __name__ == "__main__":
    main()
