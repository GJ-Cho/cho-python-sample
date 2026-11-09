"""장면 데이터 로딩 (PLAN.md 6장 core/loader.py).

- load_zdf: Zivid ZDF 파일 → SceneData. **호출 측에서 zivid.Application()을
  활성화한 상태여야 한다** (Zivid Application은 프로세스 단위 리소스이므로
  이 모듈에서 생성하지 않는다). 사용 예는 scripts/inspect_zdf.py 참조.
- save_npz / load_npz: SDK가 없는 환경(웹 채팅 협업 루프)에서도 장면을 주고받을 수
  있도록 rgb/xyz/snr을 npz로 저장·복원한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_zdf(path: str | Path) -> "SceneData":
    """ZDF 파일을 읽어 SceneData로 변환한다.

    주의: 이 함수를 호출하기 전에 `with zivid.Application():` 컨텍스트가
    활성화되어 있어야 한다.
    """
    import zivid  # 지연 import — SDK 없는 환경에서도 npz 경로는 동작하도록

    from .types import SceneData

    path = Path(path)
    frame = zivid.Frame(path)
    point_cloud = frame.point_cloud()

    # PLAN 2장 데이터 계약. rgba_srgb는 (H,W,4) uint8 → 앞 3채널만 사용.
    rgb = np.ascontiguousarray(point_cloud.copy_data("rgba_srgb")[:, :, :3])
    xyz = point_cloud.copy_data("xyz")  # (H,W,3) float, mm, NaN
    snr = point_cloud.copy_data("snr")  # (H,W) float
    normals = point_cloud.copy_data("normals")  # (H,W,3) float, [-1,1], NaN

    return SceneData(rgb=rgb, xyz=xyz, snr=snr, normals=normals)


def save_npz(scene: "SceneData", path: str | Path) -> Path:
    """SceneData를 압축 npz로 저장한다 (SDK 없는 환경 공유용)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"rgb": scene.rgb, "xyz": scene.xyz}
    if scene.snr is not None:
        arrays["snr"] = scene.snr
    if scene.normals is not None:
        arrays["normals"] = scene.normals
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path) -> "SceneData":
    """save_npz로 저장한 npz를 SceneData로 복원한다."""
    from .types import SceneData

    data = np.load(Path(path))
    snr = data["snr"] if "snr" in data.files else None
    normals = data["normals"] if "normals" in data.files else None
    return SceneData(rgb=data["rgb"], xyz=data["xyz"], snr=snr, normals=normals)


def load(path: str | Path) -> "SceneData":
    """확장자로 로더를 분기한다 (.zdf → Zivid, .npz → numpy).

    주의: .zdf는 호출 측에서 zivid.Application()이 활성화된 상태여야 한다.
    Application 관리까지 자동으로 처리하려면 load_any()를 사용할 것.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".zdf":
        return load_zdf(path)
    if suffix == ".npz":
        return load_npz(path)
    raise ValueError(f"지원하지 않는 확장자: {suffix} (지원: .zdf, .npz)")


def load_any(path: str | Path) -> "SceneData":
    """load()와 같되, .zdf일 때 zivid.Application()을 자동으로 열고 닫는다.

    한 스크립트에서 장면을 한 번만 로드하는 배치 용도에 적합하다.
    """
    path = Path(path)
    if path.suffix.lower() == ".zdf":
        import zivid
        with zivid.Application():
            return load_zdf(path)
    return load(path)
