"""세그멘테이션 방법 공통 인터페이스 (PLAN.md 2장, 6장).

m1/m2/m3는 모두 이 Segmenter를 공유한다. 한 방법의 구현이 다른 방법 코드에
침범하지 않게 한다. 반환은 gripper-agnostic한 PickCandidate 리스트다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Segmenter(ABC):
    """장면 → 피킹 후보 리스트.

    Attributes:
        requires_xyz: 예측에 3D(xyz)가 필요한지 여부.
    """

    requires_xyz: bool = False

    def build(self) -> None:
        """모델 로드/워밍업. 사이클 타임 측정에서 제외하려면 predict 전에 1회 호출한다.

        기본은 no-op이다. predict()는 build()가 호출되지 않아도 동작해야 한다
        (내부에서 지연 빌드). 재호출은 무해해야 한다.
        """

    @abstractmethod
    def predict(self, scene) -> list:
        """SceneData를 받아 PickCandidate 리스트를 반환한다 (score 내림차순)."""
        raise NotImplementedError
