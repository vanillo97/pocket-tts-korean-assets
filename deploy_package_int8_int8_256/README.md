# pocket-tts 한국어 TTS — 배포 패키지 (deploy_package_int8_int8_256)

OpenVINO 기반 순수 CPU 음성 합성 독립 배포 패키지.
`torch`, HuggingFace 의존성 없이 실행됩니다 (음성 합성 및 화자 등록 모두 지원).

## 모델 구성 사양

- **Context max_len**: 256
- **Step Backbone**: INT8 (Weight-compressed) (~289MB)
- **Sampler (Flow + EOS)**: INT8 (Weight-compressed) (~9MB)
- **Mimi Decoder**: FP32 (2048ctx) (~20MB)
- **전체 패키지 크기**: 약 421MB

```
deploy_package_int8_int8_256/
├── README.md          패키지 안내 및 사양
├── MANUAL.md          상세 사용 가이드 및 화자 변경법
├── synth.py           음성 합성 진입점 (문장 분할·청킹 자동 지원)
├── models/            OpenVINO IR 모델
│   ├── step.xml/bin     Backbone Step (INT8 (Weight-compressed))
│   ├── sampler.xml/bin  Sampler Flow + EOS (INT8 (Weight-compressed))
│   └── mimi.xml/bin     Mimi Audio Decoder (FP32 (2048ctx))
├── assets/
│   ├── voice_cache.npz  Voice prefill 캐시 (256ctx, 화자 고정)
│   ├── cond_embed.npy   텍스트 임베딩 테이블 (16MB)
│   ├── tokenizer.model  SentencePiece 한국어 토크나이저 (62KB)
│   └── params.json      모델 하이퍼파라미터
└── tools/
    ├── enroll_ov.py     새로운 화자 등록 도구 (torch 불필요)
    ├── enroll_voice.py  화자 등록 예비 도구 (torch 필요시 사용)
    ├── mimi_enc.xml/bin Mimi 오디오 인코더 (화자 등록용)
    └── bos_voice.npy    Voice BOS 임베딩
```

## 시스템 요구사항

```bash
pip install openvino numpy scipy sentencepiece
```

- OS: Linux x86_64, Python 3.10+
- PyTorch 불필요 (`synth.py`는 torch 없이 순수 OpenVINO/NumPy로 구동)

## 사용법

```bash
cd deploy_package_int8_int8_256

# 기본 문장 합성
python synth.py --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav

# 긴 문장 (자동 문장 분할 및 청크 처리)
python synth.py --text "안녕하세요. 날씨가 정말 좋습니다. 오늘도 좋은 하루 되세요." --out long.wav --seed 1
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--text` | (기본 문장) | 합성할 텍스트 (한국어) |
| `--out` | `out.wav` | 출력 WAV 파일 경로 (24kHz Mono) |
| `--seed` | `0` | 난수 시드 (동일 시드 = 동일 음성 재현) |
| `--pkg` | (스크립트 경로) | 배포 패키지 디렉토리 경로 |

## 화자 변경 (새로운 음성 등록)

`tools/enroll_ov.py`를 사용하여 torch 없이 OpenVINO만으로 새로운 화자를 등록할 수 있습니다:

```bash
python tools/enroll_ov.py --voice /path/to/my_voice.wav --pkg .
```

- 권장: 10초 이내 깨끗한 16-bit WAV 파일
- 실행 후 `assets/voice_cache.npz`와 `assets/params.json`이 자동 갱신됩니다.
