"""구간별 시간 측정 (벤치마크용, 결과에는 영향 없음).

GPU 연산은 비동기라 그냥 time.perf_counter()를 찍으면 커널이 끝나기 전 시각이 잡힌다.
그래서 구간을 끊을 때마다 torch.cuda.synchronize()로 GPU 작업이 끝나기를 기다린다.
기록은 stats[f"t_{name}_s"]에 남긴다 (Segmenter.stats와 같은 dict를 넘기면 된다).
"""

from __future__ import annotations

import time


class StageTimer:
    def __init__(self, stats: dict, device: str = "cpu"):
        self.stats = stats
        self.sync = None
        if str(device).startswith("cuda"):
            import torch

            if torch.cuda.is_available():
                self.sync = torch.cuda.synchronize
        self._last = self._now()

    def _now(self) -> float:
        if self.sync is not None:
            self.sync()
        return time.perf_counter()

    def mark(self, name: str) -> None:
        """직전 mark(또는 생성) 이후 경과 시간을 t_<name>_s 로 기록한다."""
        now = self._now()
        self.stats[f"t_{name}_s"] = round(now - self._last, 4)
        self._last = now
