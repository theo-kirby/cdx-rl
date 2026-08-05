#!/usr/bin/env python
"""Render episodes to frames. Runs under the TRAINER interpreter.

The third far side of :mod:`harness.trainer_venv`'s bridge, after
``_episodes.py`` and ``_steps.py``, and the only one that opens a renderer.
It reads one JSON job spec on stdin and writes **raw RGB24 frames** on
stdout, back to back, which :mod:`harness.capture` pipes straight into
ffmpeg. The per-episode summary goes to the file the spec names, so that
nothing but pixels ever touches stdout.

Job spec::

    {"schema": "cdxrl-capture-job-v1",
     "module_dir": "…/src/Mod/cadex",
     "model_xml": "<path>", "task_path": "<path>",
     "summary_path": "<path>",
     "steps_per_frame": 1, "pane": [640, 480], "tile": [2, 2],
     "variant": {"disturbance": false} | {},
     "camera": {"track": "pelvis"|null, "azimuth": 120.0, …},
     "episodes": [{"policy": "<path>", "seed": 0}, …]}

## Two passes, and the second one is why

The episode is played once to collect a **trace** — the pose at every frame
boundary and ``data.actuator_force`` at every control step — and rendered
afterwards from that trace. It would be simpler to render inside the episode
loop, and this does not, for two reasons that are requirements rather than
taste:

* **A tile has to pad.** Four seeds end at four different steps, and the
  short ones hold their last frame (tinted red, so a fall is never mistaken
  for the end of the clip). Nothing can know how long to pad for until every
  episode has finished.
* **The caption carries the termination reason**, which the episode does not
  know until it is over.

The replay is exact and not an approximation: ``qpos`` and ``qvel`` are the
state, ``mj_forward`` recomputes the same ``xpos``/``xquat`` the episode had,
and each seed keeps **its own model** — ADR-103 §9's randomisation is drawn
into the model in place, and ``mjVIS_INERTIA`` draws boxes sized from
``body_mass``, so rendering seed 3 against seed 0's model would show the
wrong machine.

## What you are looking at: mass, not CAD

``tasks/stand-b8/model-model.xml`` has **24 bodies and 5 geoms** — the ground
plane and four foot/toe collision pads. Every other body is inertial only. A
default render of it is a floor and a sky: measured on sb1x, the fraction of
coloured pixels is **0.0000**. The machine is invisible.

So the capture turns on the viewer's ``mjVIS_INERTIA`` flag, which draws each
body's equivalent inertia box and takes that fraction to **0.1029** at the
nominal pose. This is a **viewer flag, not a model edit**, and the difference
matters more here than it looks: adding visual geoms would move the MJCF
digest, which moves the task digest, which would make the video a video of a
different machine from the one the run trained and ``harness steps`` scored.
The instrument must not move the thing it measures.

What you see is therefore the **mass distribution** — where the machine is
heavy and how those masses swing — not the CAD solids. For the solids, see
``mechanisms/mg-legs/rollout/README.md``; that path shows real geometry and
is blocked on a training run rather than on a pipeline.

## The torque strip is the point

The bars along the bottom of each pane are ``|data.actuator_force|`` as a
fraction of ``model.actuator_forcerange``, which is the same source
``mechanisms/mg-legs/drivers/hazard15.py`` reads and **never**
``step["action"]``. Under a position action space the action is a joint angle
in degrees, and ``_episodes._torque_columns`` reports it as N·mm; a strip
built on that would be a picture of the same silent error. Red is at or above
90 % of forcerange — hazard 15's own duty threshold — so the number that
experiment 004 reports as "13.5 % of frames" is the fraction of the clip
those bars spend red.

**The pure half of this file imports nothing but numpy** and is exercised by
``harness/test_capture.py`` under cdx-rl's own interpreter, which has no
mujoco. ``harness/capture.py`` imports the same names, so the dispatcher and
the worker cannot disagree about a frame's geometry.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from typing import Any

import numpy as np

JOB_SCHEMA = "cdxrl-capture-job-v1"

#: The viewer flags the capture turns on, by name. Recorded in the sidecar so
#: that a video and a still of the same pose can be told apart later.
VIEWER_FLAGS = ("mjVIS_INERTIA",)

#: The headlight the picture is lit by, and **the one thing this driver sets
#: on the model object rather than on the viewer**.
#:
#: `stand-b8`'s MJCF declares **no lights at all** (``model.nlight`` is 0), so
#: everything is lit by MuJoCo's default headlight at ``ambient 0.1, diffuse
#: 0.4`` — which renders this mechanism as a dark red silhouette on black,
#: legible enough to prove it works and not enough to watch.
#:
#: ``model.vis`` is the **visualiser's** block: it is read by ``mjv_*`` and
#: ``mjr_*`` and by nothing in ``mj_step``. Raising it changes no force, no
#: pose, no digest and no number in the sidecar — the MJCF on disk is
#: untouched and is still what the episode was played against. It is
#: recorded in the sidecar all the same, because a reader comparing two
#: clips should never have to wonder whether a brightness difference came
#: from the machine.
HEADLIGHT = {"ambient": 0.32, "diffuse": 0.85, "specular": 0.30}

#: Bars at or above this fraction of forcerange are drawn red. Hazard 15's
#: own threshold — ``hazard15.py`` counts the frames above 90 % and calls the
#: fraction the duty cycle — so the strip and the table are the same rule.
BRACING_FRACTION = 0.90

#: …and this is where amber starts. No threshold in the method rests on it;
#: it exists so that a motor working hard and a motor idling are not the same
#: colour, and it is stated here rather than buried in a branch.
WORKING_FRACTION = 0.50


class CaptureError(RuntimeError):
    """The capture cannot be set up as asked. Usage, not infrastructure."""


# ---------------------------------------------------------------------------
# The pure half. No mujoco, no engine, no filesystem.
# ---------------------------------------------------------------------------


def frame_divisors(control_hz: int) -> list[int]:
    """Every frame rate that lands a whole number of control steps a frame."""

    return [value for value in range(1, int(control_hz) + 1)
            if not int(control_hz) % value]


def steps_per_frame(control_hz: int, fps: int) -> int:
    """Control steps a frame, or a refusal naming the rates that would work.

    **``rollout_policy``'s rule, not a new one** (``CadexDynamics:8106``): a
    frame is taken at a control-step boundary the same way a simulation's
    frame is taken at a solver-step boundary, so ``control_hz % fps`` must be
    zero. The engine measured what breaking it does — 50 Hz sampled at 60 fps
    puts frames 1, 2 and 4 between two different actions — and this driver
    refuses rather than reproducing it in a second place.
    """

    control_hz = int(control_hz)
    fps = int(fps)
    if fps < 1 or control_hz % fps:
        divisors = ", ".join(str(value) for value in frame_divisors(control_hz))
        raise CaptureError(
            f"{fps} frames a second from a task that acts at {control_hz} Hz "
            f"is not a whole number of control steps a frame. A frame between "
            f"two actions would show a pose no action produced. Rates that "
            f"divide {control_hz} Hz: {divisors}."
        )
    return control_hz // fps


def tile_layout(count: int) -> tuple[int, int]:
    """``(columns, rows)`` for ``count`` panes — as square as it goes.

    1→1×1, 2→2×1, 4→2×2, 6→3×2. The 2×2 that ``--tile`` was asked for is the
    four-seed case, and the rest fall out rather than being special-cased.
    """

    if count < 1:
        raise CaptureError("a tile of no panes is not a video")
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    return columns, rows


def even(value: int) -> int:
    """The next even number at or below ``value``, floored at 2.

    ``yuv420p`` subsamples chroma 2×2, so libx264 refuses odd dimensions.
    Rounding *down* rather than up keeps a pane inside the offscreen
    framebuffer it was sized against.
    """

    return max(2, int(value) - (int(value) % 2))


def pane_frame(index: int, length: int) -> tuple[int, bool]:
    """Which recorded frame a pane shows at output ``index``, and if it froze.

    Past the end of a short episode the pane holds its last frame and the
    second element is true, which is what the red tint is keyed off. A tile
    whose panes simply went black at different times would read as four clips
    ending, and a tile that stretched every episode to the same length would
    be four different time bases side by side.
    """

    if length < 1:
        raise CaptureError("an episode with no frames cannot fill a pane")
    if index < length:
        return index, False
    return length - 1, True


def bar_columns(width: int, count: int, gap: int = 2) -> list[tuple[int, int]]:
    """``[(x0, x1), …]`` for ``count`` torque bars across ``width`` pixels.

    Evenly spaced with the remainder spread rather than dumped on the last
    bar: ten motors across 640 px is 64 each, and a strip whose last bar is
    four pixels wider than the rest reads as a motor with more authority.
    """

    if count < 1:
        return []
    edges = [round(index * width / count) for index in range(count + 1)]
    columns: list[tuple[int, int]] = []
    for index in range(count):
        left = edges[index] + gap
        right = edges[index + 1] - gap
        if right - left < 1:                      # too narrow to inset
            left, right = edges[index], edges[index + 1]
        columns.append((left, right))
    return columns


def bar_colour(fraction: float) -> tuple[int, int, int]:
    """Green / amber / red, on the thresholds declared at the top of the file."""

    if fraction >= BRACING_FRACTION:
        return (226, 66, 52)
    if fraction >= WORKING_FRACTION:
        return (232, 172, 42)
    return (72, 198, 112)


def caption(policy: str, iteration: int | None, seed: int, time_s: float,
            step: int, max_steps: int, worst_fraction: float,
            termination: str | None) -> tuple[str, str]:
    """The two columns ``mjr_overlay`` draws: labels, then values.

    ``termination`` is passed only on the frame that *is* the last one, so a
    clip says how it ended for exactly as long as the freeze holds and not
    from the first frame — a caption that announced ``tipped`` at t=0 would
    be reading the answer out before the episode.
    """

    labels = ["policy", "iteration", "seed", "t", "step", "worst servo"]
    values = [
        policy,
        "—" if iteration is None else f"{iteration}",
        f"{seed}",
        f"{time_s:.2f} s",
        f"{step}/{max_steps}",
        f"{worst_fraction * 100:.0f}% of forcerange",
    ]
    if termination is not None:
        labels.append("ended")
        values.append(termination or "survived the horizon")
    return "\n".join(labels), "\n".join(values)


def draw_torque_strip(frame: np.ndarray, fractions: np.ndarray,
                      height: int | None = None) -> None:
    """Per-motor load as bars along the bottom of ``frame``, in place.

    A dark band, one bar per actuator, and a white rule at
    :data:`BRACING_FRACTION` so the eye can read "above the line" without
    counting pixels. Bars are clipped at 1.0: a saturated servo is drawn full
    height and the caption carries the number.
    """

    pixels_high, pixels_wide, _ = frame.shape
    band_height = int(height or max(26, pixels_high // 11))
    band_height = min(band_height, pixels_high - 1)
    top = pixels_high - band_height
    band = frame[top:]
    band[:] = (band.astype(np.uint16) * 40 // 100).astype(np.uint8)

    inner = band_height - 6
    for (left, right), fraction in zip(bar_columns(pixels_wide, len(fractions)),
                                       fractions):
        value = float(min(max(fraction, 0.0), 1.0))
        bar = max(1, int(round(value * inner)))
        band[band_height - 3 - bar:band_height - 3, left:right] = bar_colour(
            float(fraction))
    rule = band_height - 3 - int(round(BRACING_FRACTION * inner))
    band[rule:rule + 1, :] = (235, 235, 235)


def tint_frozen(frame: np.ndarray) -> np.ndarray:
    """A held frame, reddened, so a fall never reads as the end of the clip.

    Returns a **copy**: the cached pane it is made from is shown again on
    every later frame and tinting it in place would compound.
    """

    tinted = frame.astype(np.uint16)
    tinted[..., 0] = np.minimum(255, tinted[..., 0] + 40)
    tinted[..., 1] = tinted[..., 1] * 42 // 100
    tinted[..., 2] = tinted[..., 2] * 42 // 100
    return tinted.astype(np.uint8)


def compose(panes: list[np.ndarray], columns: int, rows: int) -> np.ndarray:
    """Panes into one frame, row-major, with the empty cells left black."""

    pane_h, pane_w, _ = panes[0].shape
    canvas = np.zeros((pane_h * rows, pane_w * columns, 3), dtype=np.uint8)
    for index, pane in enumerate(panes):
        row, column = divmod(index, columns)
        canvas[row * pane_h:(row + 1) * pane_h,
               column * pane_w:(column + 1) * pane_w] = pane
    return canvas


# ---------------------------------------------------------------------------
# The half that needs the engine.
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def note(message: str) -> None:
    """Progress, on stderr. **stdout is pixels and nothing else.**"""

    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _init(spec: dict[str, Any]) -> None:
    sys.path.insert(0, spec["module_dir"])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import CadexDynamics as cd  # noqa: E402

    # The ONE spelling of "nothing pushing". `--quiet` has to mean exactly
    # what `hazard15.py` means by it or the two numbers stop being about the
    # same episode, and `_episodes.apply_variant` is where that lives.
    from _episodes import apply_variant  # noqa: E402

    task = json.loads(open(spec["task_path"], encoding="utf-8").read())
    _STATE["cd"] = cd
    _STATE["task"] = task
    _STATE["played"] = apply_variant(task, spec.get("variant") or {})
    _STATE["model_bytes"] = open(spec["model_xml"], "rb").read()


def _trace(request: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Play one episode; keep the poses to render and the forces to plot.

    The model is loaded here and **kept**, because it is an output: ADR-103
    §9's randomisation is multiplied into it in place, so this object is the
    machine this seed actually ran, and the render pass needs exactly it.
    """

    cd = _STATE["cd"]
    task = _STATE["task"]
    stride = int(spec["steps_per_frame"])

    container = cd.decode_policy(open(request["policy"], "rb").read(),
                                 context=os.path.basename(request["policy"]))
    header, weights = container["header"], container["weights"]

    # The POLICY's channel list, not the task's — `_steps._play`'s reason
    # applies unchanged: a header names the inputs its first layer has, and
    # once a task gains a channel the two stop being equal.
    names = [str(n) for n in header.get("observations") or ()]
    if not names:
        names = [c for row in task["observations"] for c in row["channels"]]

    model = cd.load_model(_STATE["model_bytes"])
    played = _STATE["played"]

    poses: list[dict[str, Any]] = []
    forces: list[np.ndarray] = []

    def act(_step: int, observation: Any) -> Any:
        return cd.policy_forward(header, weights,
                                 [observation[n] for n in names])

    def sample(step: int, data: Any, final: bool, _action: Any) -> None:
        # Force at EVERY control step, whatever the frame rate. Ten floats a
        # step is free, and it is what makes the sidecar's peak the same
        # measurement `hazard15.py` reports rather than a subsample of it.
        forces.append(np.abs(np.asarray(data.actuator_force, dtype=float)))
        # `rollout_policy`'s frame rule, and its last clause matters: the
        # final state is always kept however the episode ended, or a clip
        # would stop at the previous frame boundary and show a mechanism that
        # had not yet fallen over.
        if int(step) % stride and not final:
            return None
        poses.append({
            "step": int(step),
            "qpos": np.asarray(data.qpos, dtype=float).copy(),
            "qvel": np.asarray(data.qvel, dtype=float).copy(),
            "time": float(data.time),
        })
        return None

    episode = cd.evaluate_episode(model, played, actions=act, sample=sample,
                                  seed=int(request["seed"]))

    limits = np.asarray(model.actuator_forcerange[:, 1], dtype=float) * 1000.0
    stacked = np.vstack(forces) * 1000.0            # N·m → N·mm
    fractions = stacked / np.where(limits > 0.0, limits, 1.0)
    return {
        "model": model,
        "poses": poses,
        "fractions": fractions,
        "limits": limits,
        "summary": {
            "policy": os.path.basename(request["policy"]),
            "iteration": request.get("iteration"),
            "seed": int(request["seed"]),
            "step_count": int(episode["step_count"]),
            "max_steps": int(_STATE["task"]["episode"]["max_steps"]),
            "total_reward": float(episode["total_reward"]),
            "survived": bool(episode["truncated"]),
            "termination": (None if episode["truncated"]
                            else str(episode["termination"])),
            "frames": len(poses),
            "force_samples": int(stacked.shape[0]),
            "forcerange_nmm": [round(float(v), 3) for v in limits],
            "peak_nmm_per_motor": [round(float(v), 3)
                                   for v in stacked.max(axis=0)],
            "peak_frac_of_forcerange": round(float(fractions.max()), 4),
            "frac_above_90pct_worst": round(
                float((fractions >= BRACING_FRACTION).mean(axis=0).max()), 5),
        },
    }


def _camera(cd: Any, model: Any, data: Any, pose: dict[str, Any],
            settings: dict[str, Any]) -> Any:
    """A fixed world camera, or one tracking a named body.

    Fixed by default, deliberately: **a static frame is what makes a stride
    legible**, because the floor stops moving and the foot does not. Tracking
    is the flag, not the default.

    "Fixed" still has to be aimed somewhere, and ``model.stat.center`` is the
    wrong place: it averages the mechanism with a ground plane, which on this
    model puts the lookat at 0.120 m and leaves the machine small and high in
    frame. So the lookat is the centre of the **bodies' own bounding box at
    the reset pose** — a fixed point in the world, computed from the
    episode's own first frame, and mechanism-independent in a way that
    naming ``pelvis`` would not be. Experiment 000's pendulum has to render
    too.
    """

    mujoco = cd._mujoco_module()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)

    data.qpos[:] = pose["qpos"]
    data.qvel[:] = pose["qvel"]
    mujoco.mj_forward(model, data)
    # From body 1: body 0 is the world, which sits at the origin and would
    # drag the box towards it for no reason.
    positions = np.asarray(data.xpos[1:], dtype=float)
    camera.lookat[:] = (positions.min(axis=0) + positions.max(axis=0)) / 2.0

    body = settings.get("track")
    if body:
        index = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body)))
        if index < 0:
            available = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                         for i in range(int(model.nbody))]
            raise CaptureError(
                f"--track names a body {body!r} this model does not have. "
                f"It has: {', '.join(str(n) for n in available)}."
            )
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = index
    if settings.get("distance"):
        camera.distance = float(settings["distance"])
    else:
        camera.distance = float(settings.get("distance_scale") or 2.2) * float(
            model.stat.extent)
    if settings.get("azimuth") is not None:
        camera.azimuth = float(settings["azimuth"])
    if settings.get("elevation") is not None:
        camera.elevation = float(settings["elevation"])
    return camera


def _render_pane(cd: Any, seat: dict[str, Any], index: int) -> np.ndarray:
    """One pane: replay the pose, render it, caption it, plot the load."""

    mujoco = cd._mujoco_module()
    model, data = seat["model"], seat["data"]
    renderer, pose = seat["renderer"], seat["poses"][index]

    data.qpos[:] = pose["qpos"]
    data.qvel[:] = pose["qvel"]
    data.time = pose["time"]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=seat["camera"],
                          scene_option=seat["option"])

    summary = seat["summary"]
    last = index == len(seat["poses"]) - 1
    fractions = seat["fractions"][min(pose["step"],
                                      seat["fractions"].shape[0] - 1)]
    labels, values = caption(
        summary["policy"], summary.get("iteration"), summary["seed"],
        pose["time"], pose["step"], summary["max_steps"],
        float(fractions.max()),
        (summary["termination"] or "survived the horizon") if last else None,
    )

    # `Renderer.render()` renders and reads in one call, which leaves nowhere
    # to put an overlay. The three calls below are exactly what it does, with
    # `mjr_overlay` between them — the caption is drawn by MuJoCo's own text
    # renderer into the same framebuffer rather than composited afterwards.
    renderer._gl_context.make_current()
    mujoco.mjr_render(renderer._rect, renderer.scene, renderer._mjr_context)
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                       mujoco.mjtGridPos.mjGRID_TOPLEFT,
                       renderer._rect, labels, values, renderer._mjr_context)
    pixels = np.empty((renderer.height, renderer.width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(pixels, None, renderer._rect, renderer._mjr_context)
    pixels = np.flipud(pixels).copy()          # OpenGL reads bottom-up

    draw_torque_strip(pixels, fractions)
    return pixels


def run(spec: dict[str, Any]) -> dict[str, Any]:
    """Play every episode, then stream the composed frames to stdout."""

    cd = _STATE["cd"]
    mujoco = cd._mujoco_module()

    pane_w, pane_h = (int(v) for v in spec["pane"])
    columns, rows = (int(v) for v in spec["tile"])

    option = mujoco.MjvOption()
    for name in VIEWER_FLAGS:
        option.flags[getattr(mujoco.mjtVisFlag, name)] = True

    seats: list[dict[str, Any]] = []
    try:
        for request in spec["episodes"]:
            traced = _trace(request, spec)
            model = traced["model"]
            # The offscreen framebuffer is a property of the MJCF
            # (``<visual><global offwidth=…>``) and MuJoCo's default is
            # 640x480. Refused rather than raised, because the fix a reader
            # would reach for — editing the model — is the one thing this
            # driver must not do.
            if (pane_w > int(model.vis.global_.offwidth)
                    or pane_h > int(model.vis.global_.offheight)):
                raise CaptureError(
                    f"a {pane_w}x{pane_h} pane does not fit this model's "
                    f"offscreen framebuffer "
                    f"({int(model.vis.global_.offwidth)}x"
                    f"{int(model.vis.global_.offheight)}). Lower --width and "
                    "--height, or tile more panes — do NOT add a <global "
                    "offwidth> clause to the MJCF: the digest it moves is the "
                    "one the policy and the task are bound by."
                )
            for channel, value in HEADLIGHT.items():
                getattr(model.vis.headlight, channel)[:] = value
            data = mujoco.MjData(model)
            seats.append({
                **traced,
                "data": data,
                "renderer": mujoco.Renderer(model, pane_h, pane_w),
                "camera": _camera(cd, model, data, traced["poses"][0],
                                  spec.get("camera") or {}),
                "option": option,
                "cache": None,
            })
            note(f"  seed {request['seed']}: "
                 f"{traced['summary']['step_count']} control steps, "
                 f"{traced['summary']['frames']} frames, "
                 f"{'survived' if traced['summary']['survived'] else traced['summary']['termination']}")

        total = max(len(seat["poses"]) for seat in seats)
        out = sys.stdout.buffer
        for index in range(total):
            panes = []
            for seat in seats:
                source, frozen = pane_frame(index, len(seat["poses"]))
                if frozen:
                    if seat["cache"] is None:
                        seat["cache"] = tint_frozen(_render_pane(cd, seat, source))
                    panes.append(seat["cache"])
                else:
                    panes.append(_render_pane(cd, seat, source))
            out.write(compose(panes, columns, rows).tobytes())
        out.flush()
    finally:
        # `mujoco.Renderer.__del__` raises EGLError at interpreter shutdown.
        # Closing here makes `close()` idempotent-by-nulling and the noise
        # never reaches the driver's output; swallowing is right because a
        # failure to free a context we are about to exit out of says nothing
        # a reader could act on.
        for seat in seats:
            try:
                seat["renderer"].close()
            except Exception:  # noqa: BLE001
                pass

    return {
        "frames": total,
        "pane": [pane_w, pane_h],
        "tile": [columns, rows],
        "output": [pane_w * columns, pane_h * rows],
        "steps_per_frame": int(spec["steps_per_frame"]),
        "viewer_flags": list(VIEWER_FLAGS),
        "viewer_headlight": dict(HEADLIGHT),
        "bracing_fraction": BRACING_FRACTION,
        "mujoco": mujoco.__version__,
        "episodes": [seat["summary"] for seat in seats],
    }


def main() -> int:
    spec = json.loads(sys.stdin.readline())
    if spec.get("schema") != JOB_SCHEMA:
        note(f"bad schema {spec.get('schema')!r}")
        return 2
    try:
        _init(spec)
        summary = run(spec)
    except CaptureError as exc:
        note(f"capture: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        note(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}")
        return 1
    with open(spec["summary_path"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
