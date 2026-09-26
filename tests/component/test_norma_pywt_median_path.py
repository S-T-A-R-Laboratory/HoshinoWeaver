"""Tests for the experimental pywt-bootstrap/median-guided path."""

from types import SimpleNamespace

import numpy as np
import pytest

from hoshicore.component.norma.alignment import AlignmentResult
from hoshicore.component.norma.detection import DetectedStars
from hoshicore.component.norma.frame_align import (
    AlignmentCameraCandidate,
    AlignmentError,
    MATCHING_PATH_ASTERISM,
    align_frame_camera_model,
    solve_star_alignment,
    solve_staged_alignment,
)
from hoshicore.component.norma.matching import MatchResult
from hoshicore.component.norma.optimization import CameraOptimizationPolicy
from hoshicore.component.norma.types import CameraModel, Intrinsics


def _camera() -> CameraModel:
    return CameraModel(intrinsics=Intrinsics(
        focal_length_mm=20.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=200,
        image_height_px=120,
    ))


def _stars(count: int, offset: float = 0.0) -> DetectedStars:
    positions = np.column_stack((
        np.linspace(10.0, 190.0, count),
        np.linspace(10.0 + offset, 110.0 + offset, count),
    ))
    return DetectedStars(
        positions=positions,
        volumes=np.ones(count, dtype=np.float64),
    )


def _match(ref_geo, src_geo, count: int, high_index: bool = False):
    if high_index:
        ref_idx = np.arange(len(ref_geo.positions) - count,
                            len(ref_geo.positions), dtype=np.int32)
        src_idx = np.arange(len(src_geo.positions) - count,
                            len(src_geo.positions), dtype=np.int32)
    else:
        ref_idx = np.arange(count, dtype=np.int32)
        src_idx = np.arange(count, dtype=np.int32)
    pair_idx = np.column_stack((ref_idx, src_idx))
    return MatchResult(
        pair_idx=pair_idx,
        ref_pts=ref_geo.positions[ref_idx],
        src_pts=src_geo.positions[src_idx],
        rotation=np.eye(3),
        initial_pair_count=count,
    )


def _candidate(camera: CameraModel) -> AlignmentCameraCandidate:
    return AlignmentCameraCandidate(
        camera=camera,
        optimization_policy=CameraOptimizationPolicy(),
        init_source="test",
    )


def test_dual_path_keeps_bootstrap_and_dense_index_spaces_separate(
        monkeypatch):
    import hoshicore.component.norma.frame_align as module
    import hoshicore.component.norma.alignment as alignment_module

    camera = _camera()
    optimize_matches = []

    def fake_select(ref_geo, src_geo, ref_candidate, src_candidate,
                    bootstrap_scales, same_camera=False, **kwargs):
        assert len(ref_geo.positions) == 24
        assert len(src_geo.positions) == 24
        return (ref_geo, src_geo, ref_candidate, src_candidate,
                _match(ref_geo, src_geo, 8))

    def fake_guided(ref_geo, src_geo, alignment, max_distance_px=8.0):
        assert len(ref_geo.positions) == 32
        assert len(src_geo.positions) == 32
        return _match(ref_geo, src_geo, 7, high_index=True)

    def fake_optimize(match, ref_camera, src_camera, **kwargs):
        optimize_matches.append(match)
        return AlignmentResult(np.eye(3), ref_camera, src_camera)

    def fake_residual_p90(match, alignment):
        return 0.0

    monkeypatch.setattr(module, "_select_initial_alignment_candidate",
                        fake_select)
    monkeypatch.setattr(alignment_module, "guided_mutual_rematch",
                        fake_guided)
    monkeypatch.setattr(module, "optimize_alignment", fake_optimize)
    monkeypatch.setattr(alignment_module, "optimize_alignment", fake_optimize)
    monkeypatch.setattr(alignment_module, "_guided_rematch_residual_p90",
                        fake_residual_p90)
    monkeypatch.setattr(
        module.GeometryView,
        "features",
        property(lambda self: (_ for _ in ()).throw(
            AssertionError("dual-path solver must not read features directly"))),
    )

    result = solve_staged_alignment(
        _stars(24, offset=1.0),
        _stars(24, offset=2.0),
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=True,
        ref_refine_stars=_stars(32, offset=1.0),
        src_refine_stars=_stars(32, offset=2.0),
    )

    assert len(optimize_matches) == 2
    assert result.refine_status == "applied"
    assert len(result.bootstrap_ref.positions) == 24
    assert len(result.refine_ref.positions) == 32
    assert int(np.max(result.bootstrap_match.pair_idx)) < 24
    assert int(np.max(result.final_match.pair_idx)) >= 24


def test_staged_solver_without_refine_keeps_refine_geometry_absent(monkeypatch):
    import hoshicore.component.norma.frame_align as module

    camera = _camera()

    monkeypatch.setattr(
        module,
        "_select_initial_alignment_candidate",
        lambda ref_geo, src_geo, ref_candidate, src_candidate,
        bootstrap_scales, same_camera=False, **kwargs:
        (ref_geo, src_geo, ref_candidate, src_candidate,
         _match(ref_geo, src_geo, 8)),
    )
    monkeypatch.setattr(
        module,
        "optimize_alignment",
        lambda match, ref_camera, src_camera, **kwargs:
        AlignmentResult(np.eye(3), ref_camera, src_camera),
    )

    result = solve_staged_alignment(
        _stars(24),
        _stars(24),
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=False,
    )

    assert result.refine_status == "disabled"
    assert result.refine_ref is None
    assert result.refine_src is None
    assert result.final_match is result.bootstrap_match


def test_asterism_entry_does_not_access_median_when_refine_is_disabled(
        monkeypatch):
    import hoshicore.component.norma.frame_align as module
    from hoshicore.component.norma.geometry_view import GeometryView

    camera = _camera()
    ref_stars = _stars(24)
    src_stars = _stars(24, offset=1.0)
    ref_geo = GeometryView(ref_stars, camera)
    match = _match(ref_geo, GeometryView(src_stars, camera), 8)

    class SourceDetection:
        pywt_stars = src_stars

        @property
        def median_stars(self):
            raise AssertionError("median stars must remain lazy")

    monkeypatch.setattr(
        module.StarDetectionCache, "from_image",
        staticmethod(lambda image, star_detection_mode="auto": SourceDetection()))

    seen = {}

    def solve(ref_bootstrap_stars, src_bootstrap_stars, ref_candidate,
              src_candidate, **kwargs):
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            final_alignment=AlignmentResult(np.eye(3), camera, camera),
            final_match=match,
            bootstrap_match=match,
            refine_status="disabled",
        )

    monkeypatch.setattr(module, "solve_staged_alignment", solve)
    image = np.zeros((120, 200), dtype=np.float64)
    aligned = align_frame_camera_model(
        image, ref_geo, image, _candidate(camera), _candidate(camera),
        same_camera=False, bootstrap_scales=(1.0,), remap_map_scale=1.0,
        guided_refine=False, matching_path=MATCHING_PATH_ASTERISM)

    assert aligned.shape == image.shape
    assert seen["kwargs"]["ref_refine_stars"] is None
    assert seen["kwargs"]["src_refine_stars"] is None


def test_dual_path_selects_asterism_bootstrap_matcher(monkeypatch):
    import hoshicore.component.norma.frame_align as module

    camera = _camera()
    selected_matchers = []

    def fake_select(ref_geo, src_geo, ref_candidate, src_candidate,
                    bootstrap_scales, same_camera=False,
                    match_function=None):
        selected_matchers.append(match_function)
        return (ref_geo, src_geo, ref_candidate, src_candidate,
                _match(ref_geo, src_geo, 8))

    monkeypatch.setattr(module, "_select_initial_alignment_candidate",
                        fake_select)
    monkeypatch.setattr(
        module,
        "optimize_alignment",
        lambda match, ref_camera, src_camera, **kwargs:
        AlignmentResult(np.eye(3), ref_camera, src_camera),
    )

    solve_staged_alignment(
        _stars(24),
        _stars(24),
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=False,
        use_asterism_bootstrap=True,
    )

    assert selected_matchers == [module.match_star_pairs_asterism]


def test_detected_star_solver_is_image_free_and_uses_asterism(monkeypatch):
    import hoshicore.component.norma.frame_align as module

    camera = _camera()
    seen = {}

    def fake_select(ref_geo, src_geo, ref_candidate, src_candidate,
                    bootstrap_scales, same_camera=False,
                    match_function=None):
        seen["has_images"] = (
            hasattr(ref_geo, "image_gray"), hasattr(src_geo, "image_gray"))
        seen["matcher"] = match_function
        return (ref_geo, src_geo, ref_candidate, src_candidate,
                _match(ref_geo, src_geo, 8))

    monkeypatch.setattr(module, "_select_initial_alignment_candidate",
                        fake_select)
    monkeypatch.setattr(
        module,
        "optimize_alignment",
        lambda match, ref_camera, src_camera, **kwargs:
        AlignmentResult(np.eye(3), ref_camera, src_camera),
    )

    alignment, match = solve_star_alignment(
        _stars(24), _stars(24), _candidate(camera), _candidate(camera),
        bootstrap_scales=(1.0,), same_camera=True)

    assert seen["has_images"] == (False, False)
    assert seen["matcher"] is module.match_star_pairs_asterism
    assert len(match.pair_idx) == 8
    np.testing.assert_array_equal(alignment.rotation_ref_to_src, np.eye(3))


def test_dual_path_bootstrap_failure_does_not_fall_back_to_median(monkeypatch):
    import hoshicore.component.norma.frame_align as module

    camera = _camera()
    with pytest.raises(AlignmentError, match="Insufficient stars"):
        solve_staged_alignment(
            _stars(4),
            _stars(4),
            _candidate(camera),
            _candidate(camera),
            bootstrap_scales=(1.0,),
            guided_refine=True,
            ref_refine_stars=_stars(32),
            src_refine_stars=_stars(32),
        )


def test_dual_path_guided_failure_falls_back_to_pywt_result(monkeypatch):
    import hoshicore.component.norma.frame_align as module
    import hoshicore.component.norma.alignment as alignment_module

    camera = _camera()
    monkeypatch.setattr(
        module,
        "_select_initial_alignment_candidate",
        lambda ref_geo, src_geo, ref_candidate, src_candidate,
        bootstrap_scales, same_camera=False, **kwargs:
        (ref_geo, src_geo, ref_candidate, src_candidate,
         _match(ref_geo, src_geo, 8)),
    )
    bootstrap_alignment = AlignmentResult(np.eye(3), camera, camera)
    monkeypatch.setattr(module, "optimize_alignment",
                        lambda *args, **kwargs: bootstrap_alignment)
    monkeypatch.setattr(
        alignment_module,
        "guided_mutual_rematch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("guided failed")),
    )

    result = solve_staged_alignment(
        _stars(24),
        _stars(24),
        _candidate(camera),
        _candidate(camera),
        bootstrap_scales=(1.0,),
        guided_refine=True,
        ref_refine_stars=_stars(32),
        src_refine_stars=_stars(32),
    )

    assert result.refine_status == "failed_fallback"
    assert result.refine_error == "guided failed"
    assert result.final_alignment is bootstrap_alignment
    assert result.final_match is result.bootstrap_match
