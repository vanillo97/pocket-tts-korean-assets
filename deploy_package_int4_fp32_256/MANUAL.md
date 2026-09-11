# pocket-tts 배포 패키지 매뉴얼 (deploy_package_int4_fp32_256)

대상: `deploy_package_int4_fp32_256/` (합성·화자등록 모두 torch 불필요)

## 0. 환경 준비

```bash
pip install openvino numpy scipy sentencepiece
cd deploy_package_int4_fp32_256
```

## 1. TTS 음성 생성

### 1-1. 단일 문장 합성
```bash
python synth.py --text "안녕하세요." --out out.wav
```

### 1-2. 복수 문장 / 긴 텍스트 합성
`synth.py`는 문장 단위(`.`, `!`, `?` 등)로 텍스트를 자동 분할하고, max_len=256 한도 내에서 최적 청크로 묶어 순차 합성 후 자연스럽게 결합합니다:
```bash
python synth.py --text "안녕하세요. 오늘 날씨가 참 맑네요. 산책하기 좋은 날입니다." --out out_long.wav
```

### 1-3. 옵션 안내
- `--text`: 합성할 텍스트
- `--out`: 출력 wav 파일명 (기본값: `out.wav`)
- `--seed`: 난수 시드 (기본값: 0)
- `--pkg`: 패키지 위치 (기본값: 현재 스크립트 위치)

## 2. 화자 변경 (Voice 등록)

현재 등록된 화자는 `assets/voice_cache.npz`에 저장되어 있습니다.
torch 없이 OpenVINO 모델만으로 새 화자를 등록할 수 있습니다:

```bash
python tools/enroll_ov.py --voice /path/to/voice.wav --pkg .
```

- 형식: 16-bit WAV (모노/스테레오 무관, 샘플레이트 자동 변환)
- 길이: 10초 이내 권장
- 등록 완료 후:
```bash
python synth.py --text "안녕하세요. 새로운 목소리로 합성된 음성입니다." --out test_new_voice.wav
```

## 3. 모델 사양 요약
- Step Backbone: INT4 (Weight-compressed, group=32)
- Sampler: FP32
- Mimi Decoder: FP32
- Max Context Length: 256
