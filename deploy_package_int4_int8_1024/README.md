# pocket-tts 한국어 TTS — 배포 패키지

원본 1.3GB 모델을 품질 유지하면서 **347MB**로 줄인 전체 OpenVINO 합성 패키지.
`torch`, HuggingFace 캐시 없이 동작합니다 (합성·화자 등록 모두).
긴 텍스트는 문장 단위로 자동 분할하여 합성합니다.

## 구성

```
deploy_package/
├── README.md          이 파일
├── synth.py           합성 스크립트 (유일한 런타임 진입점)
├── models/            OpenVINO IR (CPU)
│   ├── step.xml/bin     backbone 통합 스텝 169MB (INT4, AR+prefill 겸용)
│   ├── sampler.xml/bin  flow_net + EOS 헤드 8.7MB (INT8)
│   └── mimi.xml/bin     Mimi 디코더 20MB (FP32)
├── assets/
│   ├── voice_cache.npz  voice prefill 캐시 49MB (화자 고정)
│   ├── cond_embed.npy   텍스트 임베딩 테이블 16MB
│   ├── tokenizer.model  SentencePiece 토크나이저 62KB
│   └── params.json      하이퍼파라미터
└── tools/
    ├── enroll_ov.py     화자 변경용 (torch 불필요, 기본)
    ├── enroll_voice.py  화자 변경용 (빌드타임, torch 필요, 예비)
    ├── mimi_enc.xml/bin Mimi 인코더 35MB (FP32, 등록용)
    └── bos_voice.npy    voice BOS 파라미터
```

## 요구사항

```
pip install openvino numpy scipy sentencepiece
```

- OS: Linux x86_64, Python 3.10 확인
- torch 불필요 (`synth.py`는 import 시 torch 존재를 차단합니다)

## 사용법

```bash
cd deploy_package
python synth.py --text "안녕하세요." --out out.wav
python synth.py --text "오늘 날씨가 좋네요." --out out2.wav --seed 1
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--text` | (기본 문장) | 합성할 한국어 텍스트 |
| `--out` | `out.wav` | 출력 wav (24kHz) |
| `--seed` | `0` | 샘플링 시드 (같은 seed → 같은 결과) |
| `--pkg` | (스크립트 위치) | 패키지 경로 |

실행 예시:

```
OV 로드: 1.4s
text tokens=11, max_gen=71, voice off=41
AR: 53 frames, 1.19s (22.3ms/step)
저장: out.wav (4.24s 오디오, 3.57x)
peak=0.516
```

## 화자 변경 (torch 불필요)

`voice_cache.npz`에 화자가 고정되어 있습니다. 바꾸려면:

```bash
python tools/enroll_ov.py --voice /path/to/new_voice.wav --pkg .
```

`assets/`의 voice_cache와 params.json이 갱신됩니다. (16-bit WAV, 10초 이내 권장. 초과분은 잘립니다.)

참고: `tools/enroll_voice.py`는 원본 torch 모델로 등록하는 예비 수단입니다
(1.2GB safetensors 필요). OV 등록 결과와 동급임을 검증했습니다.

## 제약

- 긴 텍스트는 문장 단위로 자동 분할됩니다. 한 청크당 약 110 토큰(한글 200자 내외),
  최대 512 latent(약 40초 오디오)까지 한 번에 처리합니다.
- 한 문장이 청크上限을 넘으면 assert로 중단됩니다. 그런 문장은 쉼표·접속사 기준으로 나누세요.
- 길이 제한: `voice 토큰(41) + 텍스트 토큰 + 생성 ≤ 1024` 스텝.
- `sampler_decode_steps=1`, `temp=0.3` 고정 (원본 기본값, `params.json` 참조).
- 긴 합성은 실시간보다 느릴 수 있습니다 (27초 오디오에 약 50초). 짧은 문장은 3배속 이상입니다.

## 품질 근거

- 컴포넌트별 결정적 오차(FP32 대비): step INT4 4.0%, sampler INT8 2.5%, mimi FP32 0%.
- 전체 INT8 양자화는 89% 오차로 탈락했습니다 (NaN sentinel이 NNCF 캘리브레이션을 오염).
- 최종 음성 QA가 FP32와 동급 범위임을 확인했습니다. FP32도 seed만 바꾸면 동등 수준으로 달라지므로,
  관측된 차이는 샘플링 분산이지 양자화 열화가 아닙니다.
- 속도: 40ms/step → 22ms/step.
