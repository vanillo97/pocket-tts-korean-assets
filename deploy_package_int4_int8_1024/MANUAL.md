# pocket-tts 배포 패키지 메뉴얼 — 화자 변경 · TTS 음성 생성

대상: `deploy_package/` (합성·등록 모두 torch 불필요)

## 0. 준비

```bash
pip install openvino numpy scipy sentencepiece
cd deploy_package
```

- Python 3.10, Linux x86_64에서 확인.
- 전체 용량 약 298MB. 원본 1.2GB 모델·HuggingFace 캐시 없이 동작합니다.

## 1. TTS 음성 생성

### 1-1. 기본

```bash
python synth.py --text "안녕하세요." --out out.wav
```

정상 출력 예시:

```
OV 로드: 1.2s
text tokens=11, max_gen=71, voice off=41
AR: 53 frames, 1.19s (22.5ms/step)
저장: out.wav (4.24s 오디오, 3.55x)
peak=0.516
```

- 출력은 24kHz mono WAV입니다. 바로 재생 가능합니다.
- `AR: N frames`가 `max_gen`과 같은 값에서 멈추면 EOS 미감지이므로 텍스트를 짧게 나누세요.

### 1-2. 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--text` | (기본 문장) | 합성할 텍스트 (한국어) |
| `--out` | `out.wav` | 출력 파일 경로 |
| `--seed` | `0` | 샘플링 시드. 같은 seed + 같은 텍스트 = 같은 결과 |
| `--pkg` | (스크립트 위치) | 패키지 경로. 복사·이동해도 동작 |

예시:

```bash
# 다른 seed로 변형 생성
python synth.py --text "오늘 날씨가 좋네요." --out take1.wav --seed 0
python synth.py --text "오늘 날씨가 좋네요." --out take2.wav --seed 1

# 여러 문장 배치
for i in 1 2 3; do
  python synth.py --text "문장 $i 입니다." --out "s$i.wav" --seed $i
done
```

### 1-3. 텍스트 길이 (자동 분할)

- 긴 글도 그대로 넣으면 됩니다. 문장 부호 기준 자동 분할 → 청크별 합성 → 이어붙이기.
- 한 청크당 약 110 토큰(한글 200자 내외), 최대 약 40초 오디오까지 처리합니다.
- 마침표 없이 200자를 넘기는 단일 문장은 assert로 중단됩니다. 쉼표·접속사 기준으로 나누세요.
- 청크 경계에서 낭독 톤이 살짝 바뀔 수 있습니다. 오디오북처럼 매끈한 연결이 필요하면
  3~4문장씩 나눠 합성하고 직접 이어붙이세요.
- 긴 합성은 실시간보다 느립니다 (27초 오디오에 약 50초 소요). 짧은 문장은 3배속 이상입니다.

## 2. 화자 변경 (voice 등록)

현재 화자는 `assets/voice_cache.npz`에 고정되어 있습니다.
새 화자 등록도 torch 없이 됩니다.

### 2-1. voice 파일 준비

- 형식: **16-bit WAV** (모노·스테레오 무관, 샘플레이트 무관 — 24kHz로 자동 변환).
- MP3 등 다른 형식은 WAV로 먼저 변환하세요.
- 길이: **10초 이내 권장** (초과분은 앞에서부터 10초만 사용).
- 요령: 조용한 환경, 평서문 2~3문장, 과한 감정·웃음 제외.

```bash
# 예: ffmpeg으로 변환 (필요 시)
ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 voice.wav
```

### 2-2. 등록 실행

```bash
python tools/enroll_ov.py --voice /path/to/voice.wav --pkg .
```

정상 출력 예시:

```
samples=75233, voice frames=40
conditioning: (1, 40, 1024)
voice offset=41
저장: assets/voice_cache.npz (50.3MB)
params.json voice 갱신
```

- `assets/voice_cache.npz`와 `params.json`이 갱신됩니다. (되돌리려면 재등록 전 파일을 백업해 두세요.)
- 등록 직후 테스트 합성 1건으로 확인하세요:

```bash
python synth.py --text "안녕하세요. 제 목소리가 맞는지 확인해 주세요." --out check.wav
```

### 2-3. 등록이 실패할 때

| 증상 | 원인·대처 |
|---|---|
| `wave.Error` / 파일 읽기 실패 | 16-bit WAV가 아님. ffmpeg으로 변환 후 재시도 |
| `MAX_LEN 초과` (합성 시) | voice가 길어 텍스트+생성 예산 부족. voice를 짧게(5초 내외) 재등록하거나 텍스트를 나누기 |
| 목소리가 다르게 나옴 | voice 파일 자체 문제 (잡음·겹친 말). 조용한 녹음으로 교체 |

## 3. 파일 설명

```
deploy_package/
├── synth.py           합성 (유일한 런타임 진입점)
├── models/            step(169M, INT4) / sampler(8.7M, INT8) / mimi(20M, FP32)
├── assets/            voice_cache(화자) / cond_embed / tokenizer / params.json
└── tools/
    ├── enroll_ov.py     화자 등록 (기본, torch 불필요)
    ├── enroll_voice.py  화자 등록 (예비, 원본 torch 모델 필요)
    ├── mimi_enc.xml/bin 등록용 인코더 35MB
    └── bos_voice.npy    모델 파라미터
```

- `models/`·`assets/` 파일명을 바꾸면 동작하지 않습니다. (`--step` 계열 옵션은 개발용)
- 화자를 바꾸면 `assets/voice_cache.npz` + `params.json`만 달라집니다. 모델은 그대로입니다.

## 4. 품질·성능 참고

- FP32 원본 대비 결정적 오차: step 4.0%, sampler 2.5%, mimi 0%. 최종 음성은 동급 범위에서 검증.
- 속도: 약 22ms/step, 실시간 대비 약 3.5배.
- 같은 seed에서는 결과가 완전히 재현됩니다. (버그 제보 시 seed·텍스트·check.wav를 함께 남겨주세요.)
