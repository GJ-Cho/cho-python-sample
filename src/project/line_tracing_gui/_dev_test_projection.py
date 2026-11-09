"""
Offline checks for geometry.projection - no camera, no robot, no projector.

build_projector_image takes its 3D -> projector-pixel mapping as an injectable
argument, so everything downstream of that mapping can be exercised here: the
NaN and field-of-view masking, the colour a segment ends up in, and - the part
worth being strict about - that the drawn line is *broken* wherever a waypoint
could not be projected rather than bridged straight across the gap. A bridged
line would be light on the surface describing a path the waypoints never
described, which is worse than showing nothing.

What is NOT tested here is the mapping itself. Whether
zivid.projection.pixels_from_3d_points agrees with where the light actually
lands is the camera's answer to give, and only a real camera can give it.

Run:
    python -m line_tracing_gui._dev_test_projection

"""

import sys

import numpy as np

from line_tracing_gui.geometry.projection import (
    BACKGROUND_BGRA,
    LINE_THICKNESS_PX,
    VERDICT_COLORS_BGRA,
    VERDICT_OK,
    VERDICT_UNCHECKED,
    VERDICT_UNREACHABLE,
    VERDICT_WARNING,
    build_projector_image,
)

RESOLUTION = (200, 320)  # (height, width), the shape build_projector_image works in
FAKE_CAMERA = object()  # never touched: the mapping is injected


def _straight_line_mapper(scale: float = 1.0, offset=(20.0, 100.0)):
    """Map XY straight through to projector pixels, ignoring Z. Deterministic and trivially
    invertible, so a test can say exactly which pixel a point should land on."""

    def mapper(_camera, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return np.column_stack([points[:, 0] * scale + offset[0], points[:, 1] * scale + offset[1]])

    return mapper


def _mapper_with_nan(nan_indices):
    base = _straight_line_mapper()

    def mapper(camera, points):
        pixels = base(camera, points)
        pixels[list(nan_indices)] = np.nan
        return pixels

    return mapper


def _points_along_x(count: int, step: float = 10.0, y: float = 0.0):
    return np.array([[index * step, y, 500.0] for index in range(count)], dtype=np.float32)


def _drawn_mask(image):
    """True where something was drawn, i.e. the pixel is not the black background."""
    return np.any(image != np.array(BACKGROUND_BGRA, dtype=np.uint8), axis=2)


def _pure_verdict_colors(image):
    """The verdict colours actually drawn, judged on fully-saturated pixels only.

    Lines and dots are drawn antialiased, so the edges of every stroke are blends of the
    stroke colour and the black background - collecting every distinct pixel value would
    return dozens of shades and say nothing. The core of a stroke is the exact colour, so
    that is what gets counted. A blend can never be mistaken for a different verdict colour
    here: blending towards black scales all channels down together, and no verdict colour is
    a scaled-down version of another.
    """
    drawn = image[_drawn_mask(image)]
    present = {tuple(int(channel) for channel in color) for color in np.unique(drawn, axis=0)}
    return present & set(VERDICT_COLORS_BGRA.values())


def _build_single(camera, points_camera_frame, verdicts=None, **kwargs):
    """One path, spelled the way the single-path scenarios read best.

    build_projector_image always takes a list of paths; wrapping here keeps the scenarios
    below about what is being drawn rather than about list-of-one plumbing.
    """
    return build_projector_image(
        camera,
        [points_camera_frame],
        verdicts=None if verdicts is None else [verdicts],
        **kwargs,
    )


def _report(passed: bool, description: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {description}{('  - ' + detail) if detail else ''}")
    return passed


def _main() -> None:
    checks = []

    print("0. importing the module must not load cv2, and drawing must load it lazily")
    # ZividQtApplication refuses to start when cv2 is already in sys.modules ("It has
    # conflicting versions of Qt on some platforms"), and main.py imports geometry.projection
    # - via main_window and TracePanel - before constructing the application. An `import cv2`
    # at this module's top level therefore stops the whole GUI from launching. That happened
    # once; this pins it. geometry.projection is imported at the top of this file, so if it
    # pulled cv2 in, cv2 would be in sys.modules by now.
    checks.append(
        _report("cv2" not in sys.modules, "importing geometry.projection leaves cv2 unloaded")
    )

    print("1. a clean path renders, on the projector's own resolution")
    points = _points_along_x(10)
    result = _build_single(
        FAKE_CAMERA, points, resolution_height_width=RESOLUTION, pixel_mapper=_straight_line_mapper()
    )
    checks.append(
        _report(
            result.image_bgra.shape == (RESOLUTION[0], RESOLUTION[1], 4),
            "image is (height, width, 4) BGRA",
            str(result.image_bgra.shape),
        )
    )
    checks.append(_report(result.image_bgra.dtype == np.uint8, "dtype is uint8", str(result.image_bgra.dtype)))
    checks.append(
        _report(
            result.drawn_waypoint_count == 10 and result.skipped_waypoint_count == 0,
            "all 10 drawn",
            result.summary(),
        )
    )
    checks.append(_report(result.segment_count == 9, "9 segments for 10 points", f"got {result.segment_count}"))
    checks.append(_report(bool(_drawn_mask(result.image_bgra).any()), "something was actually drawn"))
    # ...and now it is loaded, by CV2Handler's own deferred import inside __init__ - which is
    # the mechanism that keeps the GUI startable.
    checks.append(_report("cv2" in sys.modules, "drawing loaded cv2 lazily, via CV2Handler"))

    print("\n2. with no reachability check, everything is the unchecked colour")
    checks.append(
        _report(
            _pure_verdict_colors(result.image_bgra) == {VERDICT_COLORS_BGRA[VERDICT_UNCHECKED]},
            "only the unchecked colour is drawn",
            str(_pure_verdict_colors(result.image_bgra)),
        )
    )

    print("\n3. verdicts drive the colours, and a segment takes the worse of its two ends")
    verdicts = [VERDICT_OK] * 10
    verdicts[5] = VERDICT_UNREACHABLE
    verdicts[8] = VERDICT_WARNING
    result = _build_single(
        FAKE_CAMERA,
        points,
        verdicts=verdicts,
        resolution_height_width=RESOLUTION,
        pixel_mapper=_straight_line_mapper(),
    )
    present = _pure_verdict_colors(result.image_bgra)
    checks.append(
        _report(
            present
            == {
                VERDICT_COLORS_BGRA[VERDICT_OK],
                VERDICT_COLORS_BGRA[VERDICT_UNREACHABLE],
                VERDICT_COLORS_BGRA[VERDICT_WARNING],
            },
            "exactly the three colours in play are drawn",
            str(present),
        )
    )
    checks.append(
        _report(
            VERDICT_COLORS_BGRA[VERDICT_UNCHECKED] not in present,
            "the unchecked colour is gone once a check has run",
        )
    )
    # Nothing is dropped, so the line is continuous: 9 segments regardless of colour changes.
    checks.append(
        _report(result.segment_count == 9, "colour changes do not break the line", f"got {result.segment_count}")
    )

    # Regression: a segment's colour must come from its own two ends. An earlier version
    # carried the run's colour forward instead, so the first unreachable waypoint turned
    # everything behind it red and hid every later verdict - the exact opposite of what this
    # display exists for. Pinned geometrically: with the path running left to right and the
    # unreachable waypoint in the middle, green must still appear to the right of all red.
    def _rightmost_x(verdict):
        found = np.argwhere(
            np.all(result.image_bgra == np.array(VERDICT_COLORS_BGRA[verdict], dtype=np.uint8), axis=2)
        )
        return int(found[:, 1].max()) if len(found) else None

    rightmost_red = _rightmost_x(VERDICT_UNREACHABLE)
    rightmost_green = _rightmost_x(VERDICT_OK)
    checks.append(
        _report(
            rightmost_red is not None and rightmost_green is not None and rightmost_green > rightmost_red,
            "an unreachable waypoint does not swallow the colours behind it",
            f"rightmost red x={rightmost_red}, rightmost green x={rightmost_green}",
        )
    )

    print("\n4. a point the projector cannot map (NaN) breaks the line, it is not bridged")
    result = _build_single(
        FAKE_CAMERA, points, resolution_height_width=RESOLUTION, pixel_mapper=_mapper_with_nan([4])
    )
    checks.append(
        _report(
            result.drawn_waypoint_count == 9 and result.skipped_waypoint_count == 1,
            "the NaN point is skipped",
            result.summary(),
        )
    )
    # 10 points minus 1 = two runs of 4 and 5 -> 3 + 4 = 7 segments, not 8.
    checks.append(
        _report(result.segment_count == 7, "the gap is not bridged (7 segments, not 8)", f"got {result.segment_count}")
    )
    checks.append(
        _report("outside the projector" in result.summary(), "summary mentions the skipped points", result.summary())
    )

    print("\n5. points outside the projector's resolution are skipped the same way")
    # Push half the line off the right-hand edge of a 320 px wide projector.
    wide_points = _points_along_x(10, step=50.0)
    result = _build_single(
        FAKE_CAMERA, wide_points, resolution_height_width=RESOLUTION, pixel_mapper=_straight_line_mapper()
    )
    expected_inside = sum(1 for index in range(10) if 0 <= index * 50.0 + 20.0 < RESOLUTION[1])
    checks.append(
        _report(
            result.drawn_waypoint_count == expected_inside,
            f"only the {expected_inside} points inside the projector image are drawn",
            result.summary(),
        )
    )
    checks.append(
        _report(
            result.skipped_waypoint_count == 10 - expected_inside,
            "the rest are counted as skipped",
            f"got {result.skipped_waypoint_count}",
        )
    )

    print("\n6. a path entirely outside the projector draws nothing and says so")
    far_points = np.array([[10_000.0, 10_000.0, 500.0]] * 5, dtype=np.float32)
    result = _build_single(
        FAKE_CAMERA, far_points, resolution_height_width=RESOLUTION, pixel_mapper=_straight_line_mapper()
    )
    checks.append(_report(result.drawn_waypoint_count == 0, "nothing drawn", result.summary()))
    checks.append(_report(not _drawn_mask(result.image_bgra).any(), "the image is left as pure background"))
    checks.append(_report("No waypoint could be projected" in result.summary(), "summary says so", result.summary()))

    print("\n7. the background is opaque black - the projector emits no light there")
    background_pixel = result.image_bgra[0, 0]
    checks.append(
        _report(
            tuple(int(channel) for channel in background_pixel) == BACKGROUND_BGRA,
            f"background is {BACKGROUND_BGRA} (BGRA)",
            str(tuple(int(channel) for channel in background_pixel)),
        )
    )

    print("\n8. bad input is refused rather than silently drawn wrong")
    for description, kwargs, expected in (
        ("no points", {"points_camera_frame": np.zeros((0, 3), dtype=np.float32)}, "No points"),
        (
            "verdict count mismatch",
            {"points_camera_frame": _points_along_x(5), "verdicts": [VERDICT_OK] * 4},
            "verdicts for",
        ),
    ):
        try:
            _build_single(
                FAKE_CAMERA,
                resolution_height_width=RESOLUTION,
                pixel_mapper=_straight_line_mapper(),
                **kwargs,
            )
            checks.append(_report(False, f"{description} raises ValueError", "nothing raised"))
        except ValueError as ex:
            checks.append(_report(expected in str(ex), f"{description} raises ValueError", str(ex)))

    print()
    print("9. separate paths are drawn separately - never bridged into one line")
    # Two five-point paths, well apart. If the gap between them were drawn, there would be
    # nine segments instead of eight - and light on the surface describing a move the robot
    # makes in the air, tip up, not along the part.
    path_a = _points_along_x(5, step=10.0, y=0.0)
    path_b = _points_along_x(5, step=10.0, y=60.0)
    result = build_projector_image(
        FAKE_CAMERA,
        [path_a, path_b],
        resolution_height_width=RESOLUTION,
        pixel_mapper=_straight_line_mapper(),
    )
    checks.append(
        _report(result.segment_count == 8, "8 segments for 2 paths of 5 (not 9)", f"got {result.segment_count}")
    )
    checks.append(_report(result.path_count == 2, "path_count is 2", f"got {result.path_count}"))
    checks.append(
        _report(result.drawn_waypoint_count == 10, "all 10 waypoints drawn", result.summary())
    )
    checks.append(_report("2 paths" in result.summary(), "summary says how many paths", result.summary()))

    # Nothing drawn on the rows between the two paths: the mapper puts path A at y=100 and
    # path B at y=160, so a bridging line would leave marks in between.
    drawn_rows = np.unique(np.argwhere(_drawn_mask(result.image_bgra))[:, 0])
    between = [row for row in drawn_rows if 100 + LINE_THICKNESS_PX < row < 160 - LINE_THICKNESS_PX]
    checks.append(
        _report(not between, "nothing is drawn in the gap between the paths", f"rows {between[:5]}")
    )

    print()
    print("10. each path carries its own verdicts")
    result = build_projector_image(
        FAKE_CAMERA,
        [path_a, path_b],
        verdicts=[[VERDICT_OK] * 5, [VERDICT_UNREACHABLE] * 5],
        resolution_height_width=RESOLUTION,
        pixel_mapper=_straight_line_mapper(),
    )
    present = _pure_verdict_colors(result.image_bgra)
    checks.append(
        _report(
            present == {VERDICT_COLORS_BGRA[VERDICT_OK], VERDICT_COLORS_BGRA[VERDICT_UNREACHABLE]},
            "one path green, the other red",
            str(present),
        )
    )
    green_rows = np.argwhere(
        np.all(result.image_bgra == np.array(VERDICT_COLORS_BGRA[VERDICT_OK], dtype=np.uint8), axis=2)
    )[:, 0]
    red_rows = np.argwhere(
        np.all(result.image_bgra == np.array(VERDICT_COLORS_BGRA[VERDICT_UNREACHABLE], dtype=np.uint8), axis=2)
    )[:, 0]
    checks.append(
        _report(
            len(green_rows) > 0 and len(red_rows) > 0 and green_rows.max() < red_rows.min(),
            "the colours stay on their own path, they do not bleed across the gap",
            f"green rows <= {green_rows.max() if len(green_rows) else '-'}, "
            f"red rows >= {red_rows.min() if len(red_rows) else '-'}",
        )
    )

    print()
    print("11. a path too short to trace contributes nothing but does not break the rest")
    result = build_projector_image(
        FAKE_CAMERA,
        [path_a, _points_along_x(1, y=60.0)],  # a stray click alongside a real line
        resolution_height_width=RESOLUTION,
        pixel_mapper=_straight_line_mapper(),
    )
    checks.append(
        _report(result.segment_count == 4, "only the real path contributes segments", f"got {result.segment_count}")
    )
    checks.append(
        _report(result.drawn_waypoint_count == 6, "its single point is still marked with a dot", result.summary())
    )

    print()
    print("12. mismatched verdict lists are refused")
    try:
        build_projector_image(
            FAKE_CAMERA,
            [path_a, path_b],
            verdicts=[[VERDICT_OK] * 5],
            resolution_height_width=RESOLUTION,
            pixel_mapper=_straight_line_mapper(),
        )
        checks.append(_report(False, "verdict list count mismatch raises ValueError", "nothing raised"))
    except ValueError as ex:
        checks.append(_report("verdict lists for" in str(ex), "verdict list count mismatch raises ValueError", str(ex)))

    print(f"\n{'ALL CHECKS PASSED' if all(checks) else 'FAILURES ABOVE'} ({sum(checks)}/{len(checks)})")
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
