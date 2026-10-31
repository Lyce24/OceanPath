"""
Tests for the attention heatmap helpers (``oceanpath.eval.attention``).

Neither a slide reader (openslide, TRIDENT) nor OpenCV is installed in the test
environment. Unreadable slides must fall back to the coordinate scatter; the
overlay on a slide thumbnail is drawn on a synthetic thumbnail handed over in
place of the reader. ``smooth_overlay`` is the only OpenCV call: when cv2 is
missing, ``gaussian_blur`` stands in SciPy's Gaussian filter (mirrored borders,
cut at 4 sigma, as OpenCV does for float images); the assertions hold for either.

Covers:
  * load_h5_coords / get_patch_size_from_h5: (1, N, 2) coords, attrs, fallbacks
  * normalize_attention: range, rank order, winsorised outliers, constant input
  * build_attention_overlay: scores land on the patch footprints, overlaps
    average, the background is NaN; smooth_overlay keeps that background
  * colormap and alpha blending: transparent background, alpha ramp and cap
  * draw_topk_boxes: outlines at the scaled patch coordinates
  * load_slide_thumbnail / discover_slides_in_dir
  * visualize_slide_attention: coordinate scatter (no slide, unreadable slide) and
    the thumbnail overlay, writing valid PNG files and the ranked top-k patches
"""

import sys
import types

import h5py
import matplotlib
import numpy as np
import pytest
from PIL import Image

from oceanpath.eval import attention
from oceanpath.eval.attention import (
    alpha_blend,
    apply_colormap_rgba,
    build_attention_overlay,
    discover_slides_in_dir,
    draw_topk_boxes,
    get_patch_size_from_h5,
    load_h5_coords,
    load_slide_thumbnail,
    normalize_attention,
    smooth_overlay,
    visualize_slide_attention,
)

WHITE = [255, 255, 255]


def _use_cv2_or_stand_in(mp: pytest.MonkeyPatch) -> None:
    """Leave a real cv2 alone; otherwise install a module with GaussianBlur only."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        from scipy import ndimage

        def gaussian_blur(src, ksize, sigmaX, sigmaY=0):
            sigma = (sigmaY or sigmaX, sigmaX)  # (rows, cols)
            return ndimage.gaussian_filter(src, sigma=sigma, mode="mirror", truncate=4.0)

        mp.setitem(sys.modules, "cv2", types.SimpleNamespace(GaussianBlur=gaussian_blur))


@pytest.fixture
def gaussian_blur(monkeypatch):
    _use_cv2_or_stand_in(monkeypatch)


def _assert_png(path) -> tuple[int, int]:
    with Image.open(path) as img:
        assert img.format == "PNG"
        size = img.size
        img.verify()
    return size


# ═════════════════════════════════════════════════════════════════════════════
# Coordinates from H5
# ═════════════════════════════════════════════════════════════════════════════


class TestH5Coords:
    def test_load_h5_coords_squeezes_and_returns_attrs(self, tmp_path):
        with h5py.File(tmp_path / "s.h5", "w") as f:
            ds = f.create_dataset("coords", data=np.array([[[0, 0], [256, 512]]], np.int64))
            ds.attrs["patch_size_level0"] = 512
        coords, attrs = load_h5_coords(str(tmp_path / "s.h5"))
        assert coords.dtype == np.int32
        np.testing.assert_array_equal(coords, [[0, 0], [256, 512]])
        assert attrs["patch_size_level0"] == 512
        with pytest.raises(KeyError, match="xy"):
            load_h5_coords(str(tmp_path / "s.h5"), h5_coord_key="xy")

    def test_patch_size_comes_from_the_first_slide_that_records_it(self, tmp_path):
        with h5py.File(tmp_path / "no_attr.h5", "w") as f:
            f["coords"] = np.zeros((2, 2), np.int32)
        (tmp_path / "corrupt.h5").write_bytes(b"not hdf5")
        for sid, size in (("first", 448), ("second", 224)):
            with h5py.File(tmp_path / f"{sid}.h5", "w") as f:
                f.create_dataset("coords", data=np.zeros((2, 2), np.int32))
                f["coords"].attrs["patch_size_level0"] = size

        order = ["no_attr", "absent", "corrupt", "first", "second"]
        assert get_patch_size_from_h5(str(tmp_path), order) == 448
        assert get_patch_size_from_h5(str(tmp_path), ["no_attr", "absent", "corrupt"]) is None
        assert get_patch_size_from_h5(str(tmp_path / "nowhere"), ["first"]) is None


# ═════════════════════════════════════════════════════════════════════════════
# Normalisation and overlay
# ═════════════════════════════════════════════════════════════════════════════


class TestNormalizeAttention:
    def test_min_max_scaling_without_clipping(self):
        scores = normalize_attention(np.array([2.0, 4.0, 3.0, 6.0]), clip_percentile=None)
        assert scores.dtype == np.float32
        np.testing.assert_allclose(scores, [0.0, 0.5, 0.25, 1.0], atol=1e-6)

    def test_range_and_rank_order(self):
        attn = np.random.default_rng(0).standard_normal(200)
        scores = normalize_attention(attn)
        assert scores.min() == 0.0
        assert scores.max() == pytest.approx(1.0, abs=1e-6)
        # Monotone: clipping can only merge the extremes, never reorder.
        assert np.all(np.diff(scores[np.argsort(attn)]) >= 0)

    def test_an_outlier_is_winsorised(self):
        attn = np.r_[np.linspace(0, 1, 999), 1e6]
        raw = normalize_attention(attn, clip_percentile=None)
        clipped = normalize_attention(attn)  # (1, 99) percentiles
        assert np.median(raw[:-1]) < 1e-5  # one outlier squashes everything else ...
        assert np.median(clipped[:-1]) == pytest.approx(0.5, abs=0.01)  # ... unless clipped
        assert clipped[-1] == pytest.approx(1.0, abs=1e-6)

    def test_constant_or_2d_input(self):
        np.testing.assert_array_equal(normalize_attention(np.full(5, 0.3)), np.zeros(5))
        scores = normalize_attention(np.array([[1.0], [3.0]]), clip_percentile=None)
        np.testing.assert_allclose(scores, [0.0, 1.0], atol=1e-6)


class TestOverlay:
    def test_scores_fill_their_patch_footprints(self):
        # 64 px patches at downsample 16 are 4 px squares on a 12 x 8 canvas.
        coords = np.array([[0, 0], [64, 64], [128, 0]])
        overlay = build_attention_overlay(np.array([0.2, 0.8, 0.5]), coords, 64, (12, 8), 16)
        expected = np.full((8, 12), np.nan, np.float32)
        expected[0:4, 0:4] = 0.2
        expected[4:8, 4:8] = 0.8
        expected[0:4, 8:12] = 0.5
        np.testing.assert_array_equal(overlay, expected)  # NaN where there is no tissue

    def test_overlapping_patches_average(self):
        coords = np.array([[0, 0], [32, 0]])  # columns 0-3 and 2-5
        overlay = build_attention_overlay(np.array([0.2, 0.6]), coords, 64, (8, 4), 16)
        expected_row = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, np.nan, np.nan]
        np.testing.assert_allclose(overlay, np.tile(expected_row, (4, 1)), rtol=1e-6)

    def test_patches_beyond_the_canvas_are_cut_or_skipped(self):
        coords = np.array([[160, 0], [512, 512]])  # straddles the right edge / fully outside
        overlay = build_attention_overlay(np.array([0.5, 0.9]), coords, 64, (12, 8), 16)
        expected = np.full((8, 12), np.nan, np.float32)
        expected[0:4, 10:12] = 0.5  # the part of the first patch that is on the canvas
        np.testing.assert_array_equal(overlay, expected)

    def test_smoothing_keeps_the_background_nan(self, gaussian_blur):
        overlay = np.full((40, 40), np.nan, np.float32)
        overlay[5:35, 5:35] = 1.0
        smoothed = smooth_overlay(overlay, sigma=1.5)
        np.testing.assert_array_equal(np.isnan(smoothed), np.isnan(overlay))
        assert smoothed[20, 20] == pytest.approx(1.0, abs=1e-5)  # uniform interior stays
        assert smoothed[5, 20] < 0.9  # the edge blends with the zero-filled background


# ═════════════════════════════════════════════════════════════════════════════
# Colour, blending and boxes
# ═════════════════════════════════════════════════════════════════════════════


class TestColourAndBlending:
    def test_background_is_transparent_and_scores_ramp_to_the_alpha_cap(self):
        overlay = np.array([[np.nan, 0.0, 1.0]], np.float32)
        rgba, mappable = apply_colormap_rgba(overlay, "inferno", transparent_bins=40, alpha_cap=0.8)
        assert rgba.shape == (1, 3, 4) and rgba.dtype == np.uint8
        assert rgba[0, 0].tolist() == [0, 0, 0, 0]  # no tissue
        assert rgba[0, 1, 3] == 0  # lowest score fades out completely
        assert rgba[0, 2, 3] == int(0.8 * 255)
        top = (np.array(matplotlib.colormaps["inferno"](1.0)[:3]) * 255).astype(np.uint8)
        np.testing.assert_array_equal(rgba[0, 2, :3], top)
        assert (mappable.norm.vmin, mappable.norm.vmax) == (0, 1)

    @pytest.mark.parametrize("as_pil", [False, True])
    def test_alpha_blend_composites_by_the_overlay_alpha(self, as_pil):
        base = np.full((1, 3, 3), 200, np.uint8)
        heat = np.array([[[9, 9, 9, 0], [0, 0, 255, 255], [0, 0, 100, 128]]], np.uint8)
        out = alpha_blend(Image.fromarray(base) if as_pil else base, heat)
        assert out.mode == "RGB"
        pixels = np.asarray(out).astype(int)
        assert pixels[0, 0].tolist() == [200, 200, 200]  # transparent: tissue untouched
        assert pixels[0, 1].tolist() == [0, 0, 255]  # opaque: heatmap colour
        a = 128 / 255
        np.testing.assert_allclose(
            pixels[0, 2], [200 * (1 - a), 200 * (1 - a), 100 * a + 200 * (1 - a)], atol=1
        )

    def test_topk_boxes_outline_the_scaled_patches(self):
        img = np.zeros((40, 40, 3), np.uint8)
        coords = np.array([[0, 0], [160, 80]])  # downsample 8 -> (0, 0) and (20, 10)
        out = draw_topk_boxes(img, np.array([1]), coords, np.array([0.1, 0.9]), 64, 8, line_width=1)
        pixels = np.asarray(out)
        assert pixels[10, 20].tolist() == WHITE  # top-left corner of patch 1's 8 px box
        assert pixels[18, 28].tolist() == WHITE  # bottom-right corner
        assert pixels[14, 20].tolist() == WHITE  # left edge
        assert not pixels[:9, :9].any()  # patch 0 is not in the top-k: no box
        assert not img.any()  # drawn on a copy


# ═════════════════════════════════════════════════════════════════════════════
# Slides on disk
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("suffix", [".svs", ".sdpc"])
def test_unreadable_slide_gives_no_thumbnail(tmp_path, suffix):
    slide = tmp_path / f"broken{suffix}"
    slide.write_bytes(b"not a slide")
    assert load_slide_thumbnail(str(slide)) == (None, None, None)


def test_discover_slides_matches_extensions_in_one_directory(tmp_path):
    for name in ("a.svs", "b.SDPC", "c.tiff", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "d.svs").write_bytes(b"")
    (tmp_path / "e.svs").mkdir()  # a directory, not a slide
    found = discover_slides_in_dir(str(tmp_path), [".svs", ".SDPC"])
    assert found == {"a": str(tmp_path / "a.svs"), "b": str(tmp_path / "b.SDPC")}
    assert discover_slides_in_dir(str(tmp_path / "nowhere"), [".svs"]) == {}


# ═════════════════════════════════════════════════════════════════════════════
# visualize_slide_attention
# ═════════════════════════════════════════════════════════════════════════════


def _lattice(n_side: int, stride: int = 256) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1) * stride


class TestCoordinateScatter:
    def test_without_a_slide_the_scatter_is_written_with_ranked_patches(self, tmp_path):
        coords = _lattice(6)
        attn = np.random.default_rng(1).random(len(coords))
        result = visualize_slide_attention(
            slide_id="s1",
            attention=attn,
            coords=coords,
            pred_class=1,
            pred_prob=0.875,
            true_label=0,
            patch_size_level0=256,
            output_dir=str(tmp_path / "out"),
            class_names={0: "benign", 1: "tumour"},
            top_k=5,
            save_dpi=30,
            n_models=3,
        )
        path = tmp_path / "out" / "s1_attention_coords.png"
        assert result["files"] == [str(path)]
        _assert_png(path)

        top = result["top_k_patches"]
        assert [p["rank"] for p in top] == [1, 2, 3, 4, 5]
        assert [p["index"] for p in top] == np.argsort(-attn)[:5].tolist()
        assert [p["score"] for p in top] == sorted((p["score"] for p in top), reverse=True)
        assert all(0.0 <= p["score"] <= 1.0 for p in top)
        for p in top:
            assert [p["coord_x"], p["coord_y"]] == coords[p["index"]].tolist()
        assert "tumour" in result["annotation"] and "benign" in result["annotation"]
        assert "3 models" in result["annotation"]

    def test_an_unreadable_slide_falls_back_to_the_scatter(self, tmp_path):
        slide = tmp_path / "s2.sdpc"
        slide.write_bytes(b"not a slide")
        result = visualize_slide_attention(
            slide_id="s2",
            attention=np.arange(9.0),
            coords=_lattice(3),
            pred_class=0,
            pred_prob=0.6,
            true_label=0,
            patch_size_level0=256,
            output_dir=str(tmp_path / "out"),
            slide_path=str(slide),
            top_k=3,
            save_dpi=30,
        )
        assert result["files"] == [str(tmp_path / "out" / "s2_attention_coords.png")]
        assert [p["index"] for p in result["top_k_patches"]] == [8, 7, 6]


class TestThumbnailOverlay:
    """Six 256 px patches in a 3 x 2 block on a 64 x 48 thumbnail at downsample 16
    (16 px per patch, tissue in x < 48, y < 32). Patch 5 at (512, 256) scores
    highest, patch 4 at (256, 256) next."""

    THUMB_RGB = (200, 180, 190)

    @pytest.fixture(scope="class")
    def rendered(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("overlay")
        thumb = Image.new("RGB", (64, 48), self.THUMB_RGB)
        slide = root / "s3.svs"
        slide.write_bytes(b"")
        coords = np.array([[0, 0], [256, 0], [512, 0], [0, 256], [256, 256], [512, 256]])
        with pytest.MonkeyPatch.context() as mp:
            _use_cv2_or_stand_in(mp)
            mp.setattr(
                attention,
                "load_slide_thumbnail",
                lambda slide_path, vis_level=2, target_size=None: (thumb, 16.0, vis_level),
            )
            result = visualize_slide_attention(
                slide_id="s3",
                attention=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 5.0]),
                coords=coords,
                pred_class=1,
                pred_prob=0.9,
                true_label=1,
                patch_size_level0=256,
                output_dir=str(root / "out"),
                slide_path=str(slide),
                top_k=2,
                save_dpi=30,
            )
        return result, root / "out"

    def test_overview_and_blended_images_are_written(self, rendered):
        result, out = rendered
        overview, blended = out / "s3_attention_overview.png", out / "s3_blended_attention.png"
        assert result["files"] == [str(overview), str(blended)]
        _assert_png(overview)
        assert _assert_png(blended) == (64, 48)  # same size as the thumbnail
        assert [p["index"] for p in result["top_k_patches"]] == [5, 4]
        # No reader can open the slide, so no top-patch crops are written.
        assert not any((out / "top_patches").iterdir())

    def test_tissue_outside_the_patches_stays_untouched(self, rendered):
        _, out = rendered
        pixels = np.asarray(Image.open(out / "s3_blended_attention.png").convert("RGB"))
        thumb = np.array(self.THUMB_RGB, np.uint8)
        assert (pixels[34:, :] == thumb).all()
        assert (pixels[:, 50:] == thumb).all()
        assert pixels[28, 44].tolist() != list(self.THUMB_RGB)  # inside the top patch
        assert pixels[16, 32].tolist() == WHITE  # corner of the top patch's box
        assert pixels[16, 16].tolist() == WHITE  # corner of the second patch's box
