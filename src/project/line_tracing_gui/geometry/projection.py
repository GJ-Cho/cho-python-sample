"""
Draw the traced line onto the real surface with the camera's built-in projector.

The 3D preview cannot catch a wrong camera-to-base chain: it maps the waypoints
back for display with the inverse of the very transform used to build them, so
the error cancels out and everything looks right (PLAN.md has a section on why
the 3D preview cannot verify this class of error). Projecting the line onto the
physical surface is the one check that does not cancel - the light lands where
the geometry says the gripper tip will go, and the eye decides whether that is
the intended place.

What a given projection actually proves depends on which pose the waypoints are
mapped back through, and the caller chooses that by what it passes as
`camera_to_base` (see TracePanel):

- **Capture pose** (`capture_pose * hand_eye`): the same chain build_waypoints
  used, so the hand-eye term cancels and only the pixel -> 3D correspondence is
  under test. One possible cause if the line is off, so the diagnosis is clean.
- **Current pose** (`current_pose * hand_eye`), with the robot moved somewhere
  else: hand_eye no longer cancels, so a correct-looking line validates the
  hand-eye chain as well. Stronger, but a wrong line no longer says which of the
  two is at fault - which is why the capture pose is the default and this is the
  follow-up.

Limits worth knowing before trusting a compensated projection from far away:

- There is no occlusion handling. `pixels_from_3d_points` is a pure projective
  mapping with no depth buffer, so a point hidden behind something from the new
  viewpoint is still projected, landing on whatever surface is actually there.
- Points the projector cannot reach come back as NaN, or as pixel coordinates
  outside its resolution. Both are dropped here, and the line is broken rather
  than bridged across the gap - a straight segment drawn over a dropped stretch
  would be a line the waypoints never described.
- The projector has a fixed focus. Moving far from the capture distance blurs
  the line even when the geometry is right.

Colours follow the convention zividsamples' own projection code uses (BGRA
tuples on an opaque black background): green for a waypoint the robot can reach,
amber for a reachability warning, red for one it cannot reach, and yellow when no
reachability check has been run - the same green/yellow pairing
hand_eye_verification_gui uses for confirmed vs unconfirmed.

**cv2 must never be imported at module scope here.** ZividQtApplication refuses
to start if cv2 is already in sys.modules ("It has conflicting versions of Qt on
some platforms"), and main.py imports this module - through main_window and
TracePanel - before it constructs the application. An `import cv2` at the top of
this file therefore stops the whole GUI from launching, which is exactly what
happened once. zividsamples' CV2Handler exists to solve this: it imports cv2
inside its own __init__, so the import happens when one is constructed rather
than when a module is loaded. Building one here, at drawing time, is well after
the application exists. Keep it that way - do not hoist it to module level, and
do not import cv2 directly.

"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import zivid

# Safe to import: this module does not pull in cv2 at its own module scope - CV2Handler
# does it inside __init__. See the note on cv2 in the module docstring above.
from zividsamples.gui.widgets.cv2_handler import CV2Handler

# Verdict per waypoint, deciding its colour. Kept as plain strings rather than an enum so
# TracePanel can hand them over without either module importing the other.
VERDICT_OK = "ok"
VERDICT_WARNING = "warning"
VERDICT_UNREACHABLE = "unreachable"
VERDICT_UNCHECKED = "unchecked"

VERDICT_COLORS_BGRA = {
    VERDICT_OK: (0, 255, 0, 255),  # green
    VERDICT_WARNING: (0, 165, 255, 255),  # amber
    VERDICT_UNREACHABLE: (0, 0, 255, 255),  # red
    VERDICT_UNCHECKED: (0, 255, 255, 255),  # yellow
}
# Worst verdict wins when two waypoints of different colours share a segment: a segment that
# ends somewhere unreachable must not be drawn as if it were fine.
VERDICT_SEVERITY = {VERDICT_OK: 0, VERDICT_UNCHECKED: 1, VERDICT_WARNING: 2, VERDICT_UNREACHABLE: 3}

BACKGROUND_BGRA = (0, 0, 0, 255)  # opaque black: the projector emits no light here
LINE_THICKNESS_PX = 3
# A dot goes on each waypoint too, so their spacing on the real surface is visible and not
# just the line through them. Its radius is CV2Handler.draw_circles' own fixed 2 px.


@dataclass
class ProjectorImageResult:
    image_bgra: np.ndarray  # HxWx4 uint8, ready for zivid.projection.show_image_bgra
    drawn_waypoint_count: int
    skipped_waypoint_count: int  # NaN, or mapped outside the projector's resolution
    segment_count: int  # line segments actually drawn (gaps break the line, so this can be < N-1)

    def summary(self) -> str:
        if self.drawn_waypoint_count == 0:
            return "No waypoint could be projected - none of them fall inside the projector's field of view."
        if self.skipped_waypoint_count:
            return (
                f"Projecting {self.drawn_waypoint_count} waypoints; {self.skipped_waypoint_count} fall outside "
                "the projector's field of view and are left out."
            )
        return f"Projecting {self.drawn_waypoint_count} waypoints."


def _valid_pixel_mask(pixels: np.ndarray, resolution_height_width: Tuple[int, int]) -> np.ndarray:
    """True where a projector pixel is usable: not NaN, and inside the projector image.

    `pixels_from_3d_points` returns (x, y) pairs, so they are checked against width first -
    the same order zividsamples' infield-correction widget uses when it reverses
    projector_resolution before masking.
    """
    height, width = resolution_height_width
    finite = ~np.isnan(pixels).any(axis=1)
    # Comparisons on NaN are False anyway, but they raise warnings; clear them out first.
    safe = np.where(np.isnan(pixels), -1.0, pixels)
    inside = (safe[:, 0] >= 0) & (safe[:, 0] < width) & (safe[:, 1] >= 0) & (safe[:, 1] < height)
    return finite & inside


def _worse(first: str, second: str) -> str:
    """The more severe of two verdicts."""
    return max((first, second), key=lambda verdict: VERDICT_SEVERITY.get(verdict, 0))


def _segment_runs(valid: np.ndarray, verdicts: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """Split the path into polylines of one colour: (colour, consecutive point indices).

    A segment's colour comes from the verdicts at *its own two ends*, taking the worse of the
    two - a segment running into an unreachable waypoint must not be drawn as if it were
    fine. Note this is deliberately per segment and not per run: carrying a run's colour
    forward would let one unreachable waypoint swallow the colour of the whole rest of the
    path behind it.

    A run breaks where a segment cannot be drawn at all - either of its ends being
    unprojectable - so the line is never bridged across a gap, and where the colour changes.
    Neighbouring runs therefore share the waypoint between them, which is what makes both
    boundary segments get drawn.
    """
    runs: List[Tuple[str, List[int]]] = []
    current_color: Optional[str] = None
    current: List[int] = []

    def flush() -> None:
        if current:
            assert current_color is not None
            runs.append((current_color, list(current)))

    for index in range(len(verdicts) - 1):
        if not (valid[index] and valid[index + 1]):
            flush()
            current_color, current = None, []
            continue
        color = _worse(verdicts[index], verdicts[index + 1])
        if current and color == current_color:
            current.append(index + 1)
            continue
        flush()
        current_color, current = color, [index, index + 1]
    flush()
    return runs


def build_projector_image(
    camera: zivid.Camera,
    points_camera_frame: np.ndarray,
    verdicts: Optional[Sequence[str]] = None,
    resolution_height_width: Optional[Tuple[int, int]] = None,
    pixel_mapper: Optional[Callable[[zivid.Camera, np.ndarray], np.ndarray]] = None,
) -> ProjectorImageResult:
    """Render the waypoint path as an image for the camera's projector.

    Args:
        camera: Connected Zivid camera whose projector will show the image.
        points_camera_frame: Nx3 waypoint positions (mm) **in the camera's frame** - i.e.
            already mapped back through whichever camera-to-base transform the caller wants
            to put under test (see the module docstring).
        verdicts: One VERDICT_* per point, deciding its colour. Defaults to all
            VERDICT_UNCHECKED, which is what should be shown before a reachability check.
        resolution_height_width: Projector resolution as (height, width). Read from the
            camera when not given.
        pixel_mapper: 3D -> projector pixel mapping. Defaults to
            zivid.projection.pixels_from_3d_points; injectable so the drawing and masking
            can be exercised without a camera (see _dev_test_projection.py).

    Raises:
        ValueError: if the point array is not Nx3, or verdicts has a different length.
    """
    points = np.asarray(points_camera_frame, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("No points to project.")
    if verdicts is None:
        verdicts = [VERDICT_UNCHECKED] * points.shape[0]
    if len(verdicts) != points.shape[0]:
        raise ValueError(f"Got {len(verdicts)} verdicts for {points.shape[0]} points.")

    if resolution_height_width is None:
        resolution_height_width = tuple(zivid.projection.projector_resolution(camera))
    if pixel_mapper is None:
        pixel_mapper = zivid.projection.pixels_from_3d_points

    pixels = np.asarray(pixel_mapper(camera, points), dtype=float).reshape(-1, 2)
    valid = _valid_pixel_mask(pixels, resolution_height_width)

    height, width = resolution_height_width
    image = np.full((height, width, len(BACKGROUND_BGRA)), BACKGROUND_BGRA, dtype=np.uint8)

    integer_pixels = np.round(np.where(np.isnan(pixels), 0.0, pixels)).astype(np.int32)
    # Constructed here rather than at import time: this is what pulls cv2 in, and it must
    # not happen until after ZividQtApplication exists (see the module docstring).
    cv2_handler = CV2Handler()

    segment_count = 0
    for color_name, indices in _segment_runs(valid, verdicts):
        cv2_handler.draw_polygons(
            image,
            integer_pixels[indices],
            color=VERDICT_COLORS_BGRA.get(color_name, VERDICT_COLORS_BGRA[VERDICT_UNCHECKED]),
            isClosed=False,
            thickness=LINE_THICKNESS_PX,
        )
        segment_count += len(indices) - 1

    # Dots take each waypoint's *own* verdict, unlike the segments around it - so the exact
    # waypoint the robot cannot reach is marked, not just the stretch leading up to it. Drawn
    # after the lines so a red dot stays visible where a green segment runs into it.
    for verdict, color in VERDICT_COLORS_BGRA.items():
        matching = [
            index for index in range(len(verdicts)) if valid[index] and verdicts[index] == verdict
        ]
        if matching:
            cv2_handler.draw_circles(image, integer_pixels[matching], circle_color=color)

    drawn = int(np.count_nonzero(valid))
    return ProjectorImageResult(
        image_bgra=image,
        drawn_waypoint_count=drawn,
        skipped_waypoint_count=int(points.shape[0] - drawn),
        segment_count=segment_count,
    )
