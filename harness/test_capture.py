"""``capture``'s geometry is pure, so it is tested without a GPU or a model.

    uv run pytest harness/test_capture.py

Every frame this driver produces is decided by arithmetic before a renderer
is opened: how many control steps a frame is, how many panes fit a grid,
which recorded frame a pane holds once its episode has ended, where a torque
bar starts and stops. All of it lives in ``harness/_capture.py``'s pure half,
which imports nothing but numpy — so it runs under cdx-rl's own interpreter,
which deliberately has no mujoco.

**Every check must be able to fail** (DESIGN's rule 6, hazard 18). Each test
below asserts both directions where there are two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import _capture, capture  # noqa: E402
from harness._capture import CaptureError  # noqa: E402
from harness.episodes import BundleError  # noqa: E402


# ---------------------------------------------------------------------------
# The frame rate rule
# ---------------------------------------------------------------------------


def test_a_frame_rate_must_divide_the_control_rate():
    """``rollout_policy``'s rule, enforced in the second place that samples.

    A frame is taken at a control-step boundary; 50 Hz at 60 fps would put
    frames 1, 2 and 4 between two different actions, which is a pose no
    action produced. The engine measured that and refuses it — so does this.
    """

    assert _capture.steps_per_frame(50, 50) == 1
    assert _capture.steps_per_frame(50, 25) == 2
    assert _capture.steps_per_frame(50, 10) == 5
    assert _capture.steps_per_frame(100, 20) == 5

    with pytest.raises(CaptureError):
        _capture.steps_per_frame(50, 60)
    with pytest.raises(CaptureError):
        _capture.steps_per_frame(50, 3)
    # …and a rate of zero or less is not a rate.
    with pytest.raises(CaptureError):
        _capture.steps_per_frame(50, 0)


def test_the_refusal_names_the_rates_that_would_work():
    """A refusal that only says no costs a round trip to find out what to pass.

    003 halved the control rate to 50 Hz, so the divisor set a reader needs
    is not the one they memorised at 100 Hz.
    """

    with pytest.raises(CaptureError) as raised:
        _capture.steps_per_frame(50, 60)
    message = str(raised.value)
    assert "1, 2, 5, 10, 25, 50" in message
    assert "60" in message

    assert _capture.frame_divisors(50) == [1, 2, 5, 10, 25, 50]
    assert _capture.frame_divisors(100) == [1, 2, 4, 5, 10, 20, 25, 50, 100]


# ---------------------------------------------------------------------------
# Tiling and padding
# ---------------------------------------------------------------------------


def test_a_tile_is_as_square_as_it_goes():
    assert _capture.tile_layout(1) == (1, 1)
    assert _capture.tile_layout(2) == (2, 1)
    assert _capture.tile_layout(3) == (2, 2)
    assert _capture.tile_layout(4) == (2, 2)      # the four-seed case
    assert _capture.tile_layout(6) == (3, 2)
    assert _capture.tile_layout(9) == (3, 3)
    # Every layout must actually hold its panes.
    for count in range(1, 26):
        columns, rows = _capture.tile_layout(count)
        assert columns * rows >= count
    with pytest.raises(CaptureError):
        _capture.tile_layout(0)


def test_dimensions_are_forced_even_and_downwards():
    """``yuv420p`` subsamples chroma 2x2 and libx264 refuses odd dimensions.

    Rounding DOWN rather than up matters: a pane is sized against the
    model's offscreen framebuffer, and rounding 639 up to 640 is fine while
    rounding 641 up to 642 asks for a buffer the model does not have.
    """

    assert _capture.even(640) == 640
    assert _capture.even(641) == 640
    assert _capture.even(639) == 638
    assert _capture.even(1) == 2      # floored, never zero
    assert _capture.even(0) == 2


def test_a_short_episode_freezes_its_last_frame_rather_than_going_black():
    """Four seeds end at four different steps, and a tile has to pad.

    A pane that went black would read as the clip ending; a pane stretched
    to the common length would be a different time base beside three others.
    So it holds its final frame, and the second element is what the red tint
    is keyed off.
    """

    assert _capture.pane_frame(0, 155) == (0, False)
    assert _capture.pane_frame(154, 155) == (154, False)
    # Past its end: the last frame, and flagged as frozen.
    assert _capture.pane_frame(155, 155) == (154, True)
    assert _capture.pane_frame(300, 155) == (154, True)
    with pytest.raises(CaptureError):
        _capture.pane_frame(0, 0)


def test_a_frozen_frame_is_reddened_and_the_original_is_not_touched():
    """The cache is shown again on every later frame; tinting it in place
    would compound until the pane was solid red."""

    frame = np.full((4, 4, 3), 120, dtype=np.uint8)
    original = frame.copy()
    tinted = _capture.tint_frozen(frame)

    assert np.array_equal(frame, original), "the source must not be mutated"
    assert tinted[..., 0].mean() > original[..., 0].mean()
    assert tinted[..., 1].mean() < original[..., 1].mean()
    assert tinted[..., 2].mean() < original[..., 2].mean()
    # Applying it twice must not run away past the top of the range.
    assert _capture.tint_frozen(tinted).max() <= 255


def test_compose_lays_panes_out_row_major_and_leaves_the_gap_black():
    panes = [np.full((3, 5, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    canvas = _capture.compose(panes, 2, 2)

    assert canvas.shape == (6, 10, 3)
    assert canvas[0, 0, 0] == 10          # top-left
    assert canvas[0, 5, 0] == 20          # top-right
    assert canvas[3, 0, 0] == 30          # bottom-left
    assert canvas[3, 5, 0] == 0           # the empty cell of a 3-of-4 tile


# ---------------------------------------------------------------------------
# The torque strip
# ---------------------------------------------------------------------------


def test_bars_share_the_width_evenly_and_never_overlap():
    """Ten motors across 640 px. A strip whose last bar is four pixels wider
    than the rest reads as a motor with more authority than the others."""

    columns = _capture.bar_columns(640, 10)
    assert len(columns) == 10
    widths = [right - left for left, right in columns]
    assert max(widths) - min(widths) <= 1
    for (_, right), (left, _) in zip(columns, columns[1:]):
        assert right <= left, "bars must not overlap"
    assert columns[0][0] >= 0 and columns[-1][1] <= 640
    assert _capture.bar_columns(640, 0) == []


def test_a_bar_that_is_too_narrow_to_inset_keeps_its_full_cell():
    """Otherwise a wide enough strip and a narrow one differ in bar COUNT,
    which is the one thing a reader counts."""

    columns = _capture.bar_columns(20, 10, gap=2)
    assert len(columns) == 10
    assert all(right > left for left, right in columns)


def test_the_bar_colour_thresholds_are_hazard_fifteens():
    """Red starts at 90 % of forcerange because that is the threshold
    ``hazard15.py`` counts frames above and calls the duty cycle. Two
    instruments, one rule."""

    assert _capture.BRACING_FRACTION == 0.90
    red = _capture.bar_colour(0.95)
    amber = _capture.bar_colour(0.60)
    green = _capture.bar_colour(0.10)
    assert red != amber != green and red != green
    # The boundaries belong to the harsher colour, in both directions.
    assert _capture.bar_colour(0.90) == red
    assert _capture.bar_colour(0.8999) == amber
    assert _capture.bar_colour(0.50) == amber
    assert _capture.bar_colour(0.4999) == green


def test_the_strip_draws_inside_the_band_and_marks_the_ninety_percent_rule():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    top_before = frame[:150].copy()
    _capture.draw_torque_strip(frame, np.array([0.0, 0.95, 0.5, 1.0]))

    assert np.array_equal(frame[:150], top_before), "the picture is not touched"
    assert frame[150:].max() > 0, "the band must have been drawn into"
    # A saturated motor is drawn full height and a silent one is not.
    band_height = max(26, 240 // 11)
    band = frame[-band_height:]
    columns = _capture.bar_columns(320, 4)
    silent = band[:, columns[0][0]:columns[0][1]]
    saturated = band[:, columns[3][0]:columns[3][1]]
    assert (saturated.max(axis=2) > 0).sum() > (silent.max(axis=2) > 0).sum()

    # The 90 % rule is drawn all the way across, including over the gaps
    # between bars — which is what lets the eye read "above the line".
    inner = band_height - 6
    rule = band_height - 3 - round(_capture.BRACING_FRACTION * inner)
    assert (band[rule] == 235).all()


def test_a_bar_past_full_scale_is_clipped_rather_than_drawn_off_the_frame():
    """A servo can report more than its own forcerange for an instant, and a
    bar taller than the band would be drawn over the picture above it."""

    full = np.zeros((240, 320, 3), dtype=np.uint8)
    _capture.draw_torque_strip(full, np.array([1.0]))
    over = np.zeros((240, 320, 3), dtype=np.uint8)
    _capture.draw_torque_strip(over, np.array([4.0]))   # 400 % of forcerange

    assert over[:150].max() == 0, "nothing may spill onto the picture"
    assert np.array_equal(full, over), "past full scale is drawn as full scale"
    # …and a bar below full scale is genuinely shorter, or the clip proves
    # nothing.
    half = np.zeros((240, 320, 3), dtype=np.uint8)
    _capture.draw_torque_strip(half, np.array([0.4]))
    assert (half.max(axis=2) > 0).sum() < (full.max(axis=2) > 0).sum()


# ---------------------------------------------------------------------------
# The caption
# ---------------------------------------------------------------------------


def test_the_caption_columns_line_up():
    labels, values = _capture.caption("stand13.001800.cxpolicy", 1800, 3,
                                      1.24, 62, 300, 0.92, None)
    assert len(labels.splitlines()) == len(values.splitlines())
    assert "stand13.001800.cxpolicy" in values
    assert "1800" in values
    assert "62/300" in values
    assert "92%" in values


def test_the_termination_is_held_only_on_the_last_frame():
    """A caption that announced ``tipped`` at t=0 would be reading the answer
    out before the episode had run."""

    _, running = _capture.caption("p", 1800, 0, 1.0, 50, 300, 0.1, None)
    assert "ended" not in running and "tipped" not in running

    labels, ended = _capture.caption("p", 1800, 0, 6.0, 300, 300, 0.1, "tipped")
    assert "ended" in labels and "tipped" in ended
    assert len(labels.splitlines()) == len(ended.splitlines())

    # Surviving the horizon is an outcome too, and it is named rather than
    # left blank — `steps` reports `termination: null` for it.
    _, survived = _capture.caption("p", 1800, 0, 6.0, 300, 300, 0.1, "")
    assert "survived the horizon" in survived


def test_a_policy_with_no_iteration_says_so_rather_than_claiming_zero():
    """``--policy FILE`` outside a run directory has no iteration, and
    ``iteration 0`` would be a claim about a checkpoint nobody made."""

    _, values = _capture.caption("p.cxpolicy", None, 0, 0.0, 0, 300, 0.0, None)
    assert "—" in values.splitlines()[1]


# ---------------------------------------------------------------------------
# Finding an encoder
# ---------------------------------------------------------------------------


def test_ffmpeg_resolution_follows_the_declared_order(monkeypatch, tmp_path):
    """``$CDX_FFMPEG`` → ``imageio_ffmpeg`` → ``PATH`` → the pixi build.

    The order runs from what this invocation asked for to what happens to be
    on this box, and it goes into the sidecar: an encoder is part of how a
    file came to look the way it does.
    """

    asked = tmp_path / "asked-for"
    asked.write_bytes(b"")
    bundled = tmp_path / "bundled"
    bundled.write_bytes(b"")
    on_path = tmp_path / "on-path"
    on_path.write_bytes(b"")
    pixi = tmp_path / "pixi"
    pixi.write_bytes(b"")

    class _Bundled:
        @staticmethod
        def get_ffmpeg_exe():
            return str(bundled)

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _Bundled)
    monkeypatch.setattr(capture.shutil, "which", lambda _name: str(on_path))
    monkeypatch.setattr(capture, "PIXI_FFMPEG", pixi)

    # 1. What the caller asked for wins over everything.
    assert capture.resolve_ffmpeg(str(asked)) == (asked, "CDX_FFMPEG")
    # 2. Then the wheel's own static binary.
    assert capture.resolve_ffmpeg(None) == (bundled, "imageio_ffmpeg")

    # 3. Then PATH — which is what a box without the wheel falls to.
    class _Broken:
        @staticmethod
        def get_ffmpeg_exe():
            raise RuntimeError("no binary for this platform")

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _Broken)
    assert capture.resolve_ffmpeg(None) == (on_path, "PATH")

    # 4. Then the pixi build. There is no ffmpeg on this box's PATH, measured
    #    2026-08-05, so this branch is the one that used to be the only one.
    monkeypatch.setattr(capture.shutil, "which", lambda _name: None)
    assert capture.resolve_ffmpeg(None) == (pixi, "pixi")

    # …and when there is nothing at all, a refusal naming all four.
    monkeypatch.setattr(capture, "PIXI_FFMPEG", tmp_path / "absent")
    with pytest.raises(CaptureError) as raised:
        capture.resolve_ffmpeg(None)
    assert "CDX_FFMPEG" in str(raised.value)


def test_an_ffmpeg_that_was_named_and_is_not_there_is_a_refusal():
    with pytest.raises(CaptureError):
        capture.resolve_ffmpeg("/nonexistent/ffmpeg")


# ---------------------------------------------------------------------------
# Resolving what to play
# ---------------------------------------------------------------------------


def _run_dir(tmp_path: Path, *iterations: int) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    for iteration in iterations:
        (run / f"stand13.{iteration:06d}.cxpolicy").write_bytes(b"")
    (run / "stand13.best.cxpolicy").write_bytes(b"")
    (run / "stand13.cxpolicy").write_bytes(b"")
    return run


def test_a_run_directory_defaults_to_its_last_checkpoint(tmp_path):
    run = _run_dir(tmp_path, 50, 100, 1800)
    assert capture.select_policies(run, [], None) == [
        (1800, run / "stand13.001800.cxpolicy")]
    assert capture.select_policies(run, [], 100) == [
        (100, run / "stand13.000100.cxpolicy")]


def test_asking_for_an_iteration_that_is_not_there_lists_the_ones_that_are(tmp_path):
    run = _run_dir(tmp_path, 50, 100, 1800)
    with pytest.raises(BundleError) as raised:
        capture.select_policies(run, [], 1750)
    assert "50, 100, 1800" in str(raised.value)


def test_the_task_bundle_is_globbed_because_the_pendulum_names_it_differently(tmp_path):
    """`stand-b8` writes ``stand-task.json`` and experiment 000's pendulum
    writes ``task-task.json``. A driver that hardcoded either would be
    mechanism-specific for no reason at all."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "task-task.json").write_text("{}")
    (run / "progress.json").write_text("{}")
    (run / "hyperparameters.json").write_text("{}")
    assert capture.find_task(run) == run / "task-task.json"

    (run / "stand-task.json").write_text("{}")
    with pytest.raises(BundleError):
        capture.find_task(run)


# ---------------------------------------------------------------------------
# The ledger — every MP4 owes a Flywheel artifact
# ---------------------------------------------------------------------------


def _payload(seeds=(0, 1), duty=(0.303, 0.136), survived=(False, True)):
    return {
        "frames": 301, "fps": 50, "encoded_fps": 50.0, "tile": [2, 1],
        "seeds": list(seeds), "variant": {},
        "task_sha256": "a" * 64, "model_sha256": "b" * 64,
        "policies": [{"name": "stand13.001800.cxpolicy", "iteration": 1800,
                      "sha256": "c" * 64}],
        "viewer_flags": ["mjVIS_INERTIA"], "bracing_fraction": 0.9,
        "mujoco": "3.10.0",
        "provenance": {"timestamp_utc": "2026-08-05T17:00:00Z",
                       "host": {"hostname": "sb1x"},
                       "cdxrl": {"commit": "d" * 40, "dirty": False}},
        "episodes": [
            {"policy": "stand13.001800.cxpolicy", "seed": seed,
             "survived": ok, "frac_above_90pct_worst": value}
            for seed, value, ok in zip(seeds, duty, survived)
        ],
    }


def _ledger_at(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "LEDGER", tmp_path / "ledger.json")


def test_the_note_says_what_the_numbers_mean_not_what_the_file_is(tmp_path):
    """`flywheel-conventions.md` §5's rule. A note reading "an mp4 of a
    policy" is a caption; what a reader needs is the duty cycle, the survival
    count, and that the shapes are inertia boxes rather than CAD solids."""

    video = tmp_path / "stand13.001800.tile.mp4"
    video.write_bytes(b"\x00" * 1024)
    item = capture.upload_item(_payload(), video, 1, None, None)

    assert item["artifact_type"] == "binary"      # there is no video type
    assert item["media_type"] == "video/mp4"
    assert "30.3 / 13.6" in item["note"]
    assert "1 of 2 survived" in item["note"]
    assert "mjVIS_INERTIA" in item["note"] and "NOT the CAD solids" in item["note"]
    # The two kinds of seed are never blurred (§3).
    assert item["metadata"]["seeds_eval"] == [0, 1]
    assert item["metadata"]["seed_trained"] == 1
    assert "seeds" not in item["metadata"] and "seed" not in item["metadata"]


def test_a_note_argument_is_prepended_rather_than_replacing_the_measurement():
    """`--note` says *why this policy*, which no driver can know. It must not
    be able to displace the numbers, which is the part that is checkable."""

    video = Path(__file__)     # any real file; only its stat and bytes matter
    plain = capture.upload_item(_payload(), video, None, None, None)
    annotated = capture.upload_item(_payload(), video, None, None, "the clamp")
    assert annotated["note"].startswith("the clamp")
    assert plain["note"] in annotated["note"]


def test_a_recorded_video_is_pending_until_it_is_marked(monkeypatch, tmp_path):
    _ledger_at(monkeypatch, tmp_path)
    video = tmp_path / "stand13.001800.tile.mp4"
    video.write_bytes(b"\x00" * 1024)
    payload = _payload()

    capture.record(video, payload, capture.upload_item(payload, video, 2, None, None))
    assert [e["upload_item"]["filename"] for e in capture.pending()] == \
        ["stand13.001800.tile.mp4"]

    assert capture.mark_published("some-node-9999") == 0
    assert capture.pending() == []
    assert capture.read_ledger()["videos"]["stand13.001800.tile.mp4"][
        "flywheel"]["node_id"] == "some-node-9999"


def test_a_re_render_owes_a_NEW_artifact(monkeypatch, tmp_path):
    """Same filename, different bytes. Keeping the old node id would claim
    the graph holds these bytes when it holds different ones."""

    _ledger_at(monkeypatch, tmp_path)
    video = tmp_path / "stand13.001800.tile.mp4"
    payload = _payload()

    video.write_bytes(b"\x00" * 1024)
    capture.record(video, payload, capture.upload_item(payload, video, 2, None, None))
    capture.mark_published("some-node-9999")
    assert capture.pending() == []

    # Re-recorded with identical bytes: still published, no new debt.
    capture.record(video, payload, capture.upload_item(payload, video, 2, None, None))
    assert capture.pending() == []

    # Re-rendered: new bytes, and the debt comes back.
    video.write_bytes(b"\x01" * 2048)
    capture.record(video, payload, capture.upload_item(payload, video, 2, None, None))
    assert len(capture.pending()) == 1


def test_a_deleted_unpublished_video_is_not_quietly_forgotten(monkeypatch, tmp_path):
    """It stops being an obligation — there are no bytes to upload — but a
    ledger that dropped it would make "every MP4 is on the graph" true by
    attrition, so ``--pending`` still names it."""

    _ledger_at(monkeypatch, tmp_path)
    video = tmp_path / "gone.mp4"
    video.write_bytes(b"\x00" * 16)
    payload = _payload()
    capture.record(video, payload, capture.upload_item(payload, video, None, None, None))
    video.unlink()

    assert capture.pending() == []
    entry = capture.read_ledger()["videos"]["gone.mp4"]
    assert entry["flywheel"]["node_id"] is None, "the record survives the file"


def test_the_output_name_says_which_conditions_produced_it(tmp_path):
    run = _run_dir(tmp_path, 1800)
    policies = capture.select_policies(run, [], None)
    assert capture.default_output(policies, False, False).name == \
        "stand13.001800.mp4"
    assert capture.default_output(policies, True, False).name == \
        "stand13.001800.tile.mp4"
    # `--quiet` is hazard 15's conditions and a different episode; the two
    # must not land on the same file.
    assert capture.default_output(policies, True, True).name == \
        "stand13.001800.quiet.tile.mp4"
