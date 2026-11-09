# synth — BlenderProc 합성 데이터 생성기 (2단계)

[BOX]·[ONE]·[FEW] 시나리오의 학습·테스트용 합성 장면을 만든다. 배경과 결정 사항은 `../research/README.md`와 `../research/data/dataset-plan.md`에 있다.

## 환경

- BlenderProc 2.8.0 (GPL-3.0). 생성된 데이터에는 GPL이 적용되지 않는다.
  - 설치 위치: `C:\Zivid\3rdparty\venv_blenderproc\`
- Blender 4.2.1: `C:\Zivid\3rdparty\blender\blender-4.2.1`
  - 설치 중 중첩 폴더 이름 변경이 `WinError 5`로 실패해서 수동으로 옮겼다.
  - 그래서 실행할 때마다 `--custom-blender-path`를 붙여야 한다.
- 미리보기와 COCO 변환은 일반 Python으로 돌린다: `C:\Zivid\3rdparty\venv_segpick\Scripts\python.exe`

## 실행

```powershell
$bp = "C:\Zivid\3rdparty\venv_blenderproc\Scripts\blenderproc.exe"
$bl = "C:\Zivid\3rdparty\blender\blender-4.2.1"
$PY = "C:\Zivid\3rdparty\venv_segpick\Scripts\python.exe"
$D  = "C:\Zivid\cho-python-sample\src\project\2d_segmentation_picking\data\synthetic"   # 절대 경로

& $bp run gen_pallet.py --custom-blender-path $bl --out "$D\box" --n 10 --seed 0            # --types 1|2|3|cycle
& $bp run gen_bin.py    --custom-blender-path $bl --out "$D\one" --mode one --n 10 --seed 0 # --classes can,flange
& $bp run gen_bin.py    --custom-blender-path $bl --out "$D\few" --mode few --n 10 --seed 0
& $PY preview.py "$D\box"                 # _preview/*.jpg + 자기검증 수치
& $PY build_coco.py "$D\box" --top-only   # coco_top.json
```

- 시드가 같으면 같은 장면이 다시 나온다. 장면 i의 난수는 `default_rng([seed, i])`로 만든다.
- 병렬로 생성하려면 프로세스마다 `--seed`를 다르게 준다.

## 출력 (장면 폴더 1개)

| 파일 | 내용 |
|---|---|
| `rgb.png` | 8bit RGB |
| `depth.png` | uint16, mm 단위, 광축 방향 z(planar). 0은 무효 |
| `normals.png` | uint8, OpenCV 카메라 좌표계, `n = png/127.5 - 1`. 카메라를 향하는 면은 z<0 |
| `inst.png` | uint16 인스턴스 id. 0은 배경(바닥·팔레트·빈) |
| `meta.json` | K, 카메라 포즈, 인스턴스별 클래스·bbox·가시 픽셀 수. [BOX]는 `top`(노출 여부)과 `layer`도 담는다 |

category_id
- [BOX]: 1~5 = B1~B5, 택배 상자 1~5호
  - 220×190×90, 270×180×150, 340×250×210, 410×310×280, 480×380×340
  - 장면마다 5종 중 1~3종을 섞는다
- [ONE]/[FEW]: 1 can, 2 paper_box, 3 bracket, 4 flange, 5 snack_bag, 6 stick_pack, 7 pouch

## 파일

| 파일 | 역할 |
|---|---|
| `config/pallet.yaml`, `config/bin.yaml` | 모든 장면 파라미터 (하드코딩 금지 규칙) |
| `gen_pallet.py` | [BOX] 장면. 카메라는 Zivid 3 XL250, 1408×1408, 실측 점군 피팅 내부 파라미터 |
| `gen_bin.py` | [ONE]/[FEW] 장면. 카메라는 Zivid 2+ MR130, 실측 점군 피팅 내부 파라미터, 빈에 물리 낙하 |
| `products.py` | 제품 7종 절차적 생성. 한 장면 안의 같은 클래스는 같은 SKU다 |
| `geometry.py` | 메시(박스, 회전체, 압출, 베개형)와 재질 |
| `textures.py` | 절차적 텍스처. 가상 디자인만 쓰고 외부 에셋은 쓰지 않는다 |
| `scene_common.py` | 렌더 설정, 카메라, 조명, 저장 |
| `preview.py` | 4분할 미리보기와 자기검증(법선↔depth 일치도, top 플래그 depth 순서) |
| `build_coco.py` | COCO JSON(비압축 RLE)으로 변환 |
| `add_sensor_noise.py` + `config/noise.yaml` | Zivid 유사 센서 노이즈. **본 생성에서는 깨끗한 원본만 저장하고, 학습 때 `apply_noise()`를 on-the-fly로 입힌다** (사용자 확정 2026-11-04). CLI는 미리보기용 |
| `noise_stats.py` | 실측 zdf와 합성 노이즈를 같은 지표로 비교한다 (노이즈 교정용) |

## 장면 기하 (사용자 확정 2026-11-03)

- **[BOX]**
  - 카메라는 지면에서 4.0m 높이에 두고 정면 하향으로 고정한다. 오프셋, 기울임, 회전은 없다.
  - 적재 최상단은 지면 기준 2.3m 이하로 둔다(팔레트 0.144m 포함). 그래서 작업거리는 1.7~4.0m다.
  - 광학 중심이 이미지 중앙에서 벗어나 있다(cx 719.5). 그래서 1.7m에서 팔레트 중심이 이미지 중앙에 오도록 카메라를 (13.3, 2.1)mm 옮긴다. 사용자 결정 (a).
  - 그 결과 2.3m까지 쌓아도 팔레트 모서리가 이미지 안에 남는다. 여유는 0.2~2.7px뿐이다.
  - 적재 높이는 1층부터 상한까지 균등분포로 뽑는다.
- **[ONE]/[FEW]**
  - 빈 깊이는 200 / 300 / 400mm를 0.6 / 0.2 / 0.2 확률로 섞는다.
  - 카메라 거리는 빈 바닥 기준으로 1193 ± 30mm를 유지한다.

## 센서 노이즈 교정 (2026-11-04)

실측 zdf 16장(XL250 2장 + 보정판 1장, MR130 00~12.zdf 13장)과 합성 파일럿을 `noise_stats.py`의 같은 지표로 비교해 `config/noise.yaml`을 맞췄다.

| 지표 | 실측 XL250 | 합성 [BOX] | 실측 MR130 | 합성 [ONE]/[FEW] |
|---|---|---|---|---|
| depth 결측 비율 | 2.8~2.9% | 평균 2.3% (0.6~9%) | 2.5~3.1% | 평균 3.4% (0.6~11%) |
| 고립 결측(≤4 px) | 0.02% | 0.02~0.03% | 0.01~0.03% | 0.05~0.08% |
| 그림자 쪽 | 가리는 물체의 왼쪽 | 왼쪽 | 왼쪽 | 왼쪽 |
| 그림자 폭 역산 기선 | 0.24 m | 0.22~0.25 m | 0.045 m | 0.03~0.05 m |
| 인접 법선 각도차 | 2.9° | 3.0° | 1.3~2.2° | 1.4~1.9° |

- 그림자: 프로젝터가 카메라와 같은 자세로 +x에 있다고 보고(평행 스테레오), 행마다 프로젝터 열 순서가 뒤집히는 픽셀을 결측으로 둔다. 이전 z-buffer 방식은 빈 벽 같은 가파른 면을 그림자로 오판해서 결측이 실측의 3~4배였다.
- Normal: depth.png가 mm 정수라서, 노이즈 depth로 법선을 다시 계산하면 계단 잡음이 실측의 3~4배가 된다. 그래서 렌더 법선에 작은 각도 흔들림만 준다.
- depth 고주파 잡음: 실측은 Zivid 필터 뒤라 0.03~0.15 mm로 mm 양자화(σ 0.29 mm)보다 작다. 그래서 고주파 잡음은 거의 넣지 않고, 저주파 성분만 넣는다.
- 경계 전체 결측은 끈다. 실측 결측은 그림자 쪽에만 있다.
- 처리 시간: CPU에서 장당 0.56 s(MR130), 0.74 s(XL250). 학습 DataLoader의 worker 여러 개로 돌린다.

## BlenderProc 함정 (소스로 확인)

- **`enable_normals_output`**: 호출 시점의 카메라 회전이 compositor 키프레임으로 굳는다. 한 프로세스에서 장면마다 카메라를 바꾸면 법선이 틀어진다. 그래서 월드 Normal pass를 저장하고 numpy에서 회전한다.
- **Normal pass**: 벡터 소켓을 Output File에 바로 연결하면 1채널 EXR이 된다. Separate/Combine Color 노드를 거쳐야 한다.
- **`enable_segmentation_output`**: 호출 시점에 있던 오브젝트에만 `pass_index`를 준다. 그래서 렌더 직전에 다시 부여한다.
- **sampler 이름**: `bproc.sampler.uniformSO3`가 맞다. `uniform_SO3`가 아니다.
- **출력 경로**: `--out`은 절대 경로로 준다.

## 알려진 한계 (3단계에서 영향 확인)

- **센서 노이즈는 근사치다** (`add_sensor_noise.py`). 위 표의 지표는 실측에 맞췄지만 다음은 근거가 약하다.
  - 저주파 depth 성분 크기 (MR130은 바닥 평면 rms 0.49 mm를 상한으로 씀, XL250은 가정값)
  - 스치는 각도 결측 확률 (실측에서 직접 분리해 재지 못함)
  - 어두운 장면(검은 빈 + 검은 제품) 결측: 상대 밝기 기준이라 결측이 많이 나온다. 실측 예가 없다
  - RGB 잡음과 노출은 교정하지 않았다. 실측 XL250 장면은 합성보다 훨씬 어둡다(밝기 30~60이 64%)
- **렌즈 왜곡은 근사다.** 핀홀로 렌더하고, 내부 파라미터는 실측 점군에 피팅한 값을 쓴다. 왜곡의 평균 효과는 들어가지만 가장자리의 휨은 재현하지 않는다. 사용자 결정 B, 2026-11-03.
- **연포장 형상**: 천 시뮬레이션 대신 노이즈로 변형한 베개형 메시다. 충돌 형상은 convex hull이다.
- **브래킷 충돌 형상**: convex hull이라 L자 안쪽에 다른 물체가 들어가지 못한다.
- **빈 깊이**: 200 mm는 가정값이다(미측정).
