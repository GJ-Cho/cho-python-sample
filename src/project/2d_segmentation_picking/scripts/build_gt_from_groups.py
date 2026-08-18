"""정답 세트 완성 — 사람이 지정한 조각 그룹을 마스크로 합친다 (Phase 5 선행).

`gen_gt_fragments.py`가 만든 조각(mask)들을 사람이 "N번+M번은 한 물체"로 묶어
알려준 결과(이 스크립트 안의 GROUPS/MISSING)를 받아, 물체별 union 마스크를
정답 세트로 저장한다. `core/plane.py`의 mask_iou를 그대로 써서 나중에 m1/m2
결과와 채점할 수 있다.

이 파일 자체가 "정답 원본"이다 — 사람이 채팅으로 불러준 번호를 그대로 옮겨
적었다. 나중에 수정할 일이 있으면 GROUPS/MISSING을 고치고 다시 실행한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_EVAL_DIR = _PROJECT_DIR / "data" / "eval"

# 물체 하나 = 조각 번호 리스트 하나. 사람이 2026-08-11 채팅으로 불러준 그대로.
GROUPS: dict[str, list[list[int]]] = {
    "01": [[5, 7], [10], [4, 6], [0, 1, 2, 3], [8, 9]],
    "02": [[5, 8], [14], [0, 1, 2, 3, 4, 12], [6, 7, 10], [9, 13]],
    "03": [[1, 2, 4], [13, 20], [6],
           [17, 21, 14, 10, 22, 8, 3, 0, 5],
           [15, 16, 7, 19, 12, 18, 11, 9]],
    "04": [[4, 7, 6], [3, 10], [15, 13, 14], [12, 16, 17], [18, 19],
           [2, 5, 9, 11, 1, 22]],
    "05": [[9, 13, 21, 3, 5], [22], [8], [7], [15, 20], [16, 27], [2], [6],
           [23, 18], [14, 26], [11, 12, 25], [1, 4, 0, 10, 19, 17, 24]],
    "06": [[26], [31], [21], [25, 34, 32], [0], [13, 11, 6], [7, 12],
           [20, 24], [3, 5, 8], [10, 9, 17], [2, 4], [14, 23, 18], [16],
           [1, 19, 33, 30, 27, 29], [28]],
    "07": [[7, 27, 11, 8, 5, 2], [16, 19], [29, 31], [12, 14],
           [49, 46, 47, 42], [43, 56], [13, 15, 17, 18], [23, 6, 9],
           [0, 1, 4, 3, 53, 10], [22, 28], [30, 45, 38], [44],
           [25, 24, 32], [39, 51], [48, 41], [50], [21], [20]],
    "08": [[15], [8], [7], [5, 2], [18, 10, 13], [6], [17, 20, 9],
           [0, 1, 26], [3, 11], [4, 12], [14], [25, 28], [16, 23],
           [27, 30, 22], [21, 24, 29]],
    "09": [[4, 3], [12, 14, 9, 8], [1, 5], [2, 6], [10, 18], [0], [13, 11, 7]],
    "10": [[0], [8], [4], [7, 11], [10, 12, 9], [6], [1, 2, 3, 5]],
    "11": [[10], [1, 7], [2, 6, 8], [5], [3], [9], [0, 11]],
    "12": [[6, 7, 8], [9], [0], [4], [3, 5], [2]],
}

# 조각이 아예 생성되지 않은(SAM2가 못 찾은) 물체 — 대략적 위치 설명만 존재.
MISSING: dict[str, list[str]] = {
    "01": [], "02": [], "03": [], "04": [],
    "05": ["우측 상단 노란색/주황색 원형 물체"],
    "06": ["아래쪽 동그란 물체 1", "아래쪽 동그란 물체 2", "아래쪽 ㄴ자 쇠 물체"],
    "07": ["우측 아래 노란 원형물체", "우측 아래 핑크색 원형 물체",
           "노란색 오리 뒷모습", "가운데 아래 원형 물체"],
    "08": ["가운데 위쪽 노란 둥근 물체"],
    "09": [], "10": [],
    "11": ["좌측 아래 노란 원형", "중앙 아래 노란 원형"],
    "12": [],
}

# 물체 단위 메모(기각 사유는 아니고 참고 정보) — {scene: {group의 첫 조각 id: 메모}}
NOTES: dict[str, dict[int, str]] = {
    "04": {2: "22번 조각은 빛 반사 영역 포함"},
    "06": {16: "아래쪽 노란 부분이 이 물체에 속하지만 조각으로 안 잡힘(부분 누락)"},
}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    summary = []
    for scene, groups in GROUPS.items():
        frag_npz = _EVAL_DIR / f"{scene}_fragments.npz"
        frag_json = _EVAL_DIR / f"{scene}_fragments.json"
        if not frag_npz.exists():
            print(f"[건너뜀] {scene}: {frag_npz} 없음", file=sys.stderr)
            continue

        frags = np.load(frag_npz)
        n_fragments = json.loads(frag_json.read_text(encoding="utf-8"))["n_fragments"]

        objects_meta = []
        gt_masks = {}
        used_fragment_ids = set()
        for oi, ids in enumerate(groups):
            used_fragment_ids.update(ids)
            union = None
            for fid in ids:
                m = frags[f"m{fid}"]
                union = m.copy() if union is None else (union | m)
            gt_masks[f"obj{oi}"] = union
            objects_meta.append({
                "obj_id": oi,
                "fragment_ids": ids,
                "area_px": int(union.sum()),
                "note": NOTES.get(scene, {}).get(ids[0]),
            })

        unused = sorted(set(range(n_fragments)) - used_fragment_ids)

        out_npz = _EVAL_DIR / f"{scene}_gt.npz"
        np.savez_compressed(out_npz, **gt_masks)
        out_json = _EVAL_DIR / f"{scene}_gt.json"
        out_json.write_text(json.dumps({
            "scene": scene,
            "n_fragments_total": n_fragments,
            "n_objects_labeled": len(groups),
            "unused_fragment_ids": unused,
            "n_missing_objects": len(MISSING.get(scene, [])),
            "missing_objects": MISSING.get(scene, []),
            "objects": objects_meta,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        n_true_objects = len(groups) + len(MISSING.get(scene, []))
        summary.append((scene, len(groups), len(MISSING.get(scene, [])), n_true_objects,
                        n_fragments, len(unused)))
        print(f"[저장] {out_npz}", file=sys.stderr)
        print(f"[저장] {out_json}", file=sys.stderr)

    print("\nscene | 라벨된 물체 | 누락 | 정답 물체(추정) | 조각수 | 미사용 조각", file=sys.stderr)
    for row in summary:
        print(f"{row[0]:>5} | {row[1]:>9} | {row[2]:>4} | {row[3]:>13} | {row[4]:>6} | {row[5]:>10}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
