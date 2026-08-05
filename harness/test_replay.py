"""``replay``'s bookkeeping is pure, so it is tested without an engine.

    uv run pytest harness/test_replay.py

Everything that decides whether a set is *trustworthy* happens before `cadexd`
is spawned: which checkpoint was chosen, which model the bundle actually names,
what the manifest claims, whether the bytes on the far machine are the bytes
that left this one, and whether a re-export invalidates a published node. All
of that is arithmetic, digests and dictionary handling, so it runs under
cdx-rl's own interpreter with no MuJoCo, no GPU and no project store.

What is **not** tested here is the build, and deliberately: `install_and_build`
is the preflight and the real build in one function precisely so that no test
has to predict the engine's answer. The engine's answer is measured against
real arms in `mechanisms/mg-legs/rollout/README.md`, and against five
deliberately mutated scripts in ADR-135.

**Every check must be able to fail** (DESIGN's rule 6, hazard 18). Each test
below asserts both directions where there are two.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import replay  # noqa: E402
from harness.episodes import BundleError  # noqa: E402
from harness.replay import ReplayError  # noqa: E402


# ---------------------------------------------------------------------------
# A fabricated run and a fabricated bundle, built forwards.
# ---------------------------------------------------------------------------


MODEL_XML = b'<mujoco model="fake"><worldbody/></mujoco>'


def _bundle(model_sha: str | None = None) -> dict:
    return {
        "schema": "cadex-training-task-v1",
        "label": "stand",
        "episode": {"max_steps": 300, "control_hz": 50},
        "actions": [{"actuator": "a", "index": 0, "unit": "deg",
                     "low": -25.0, "high": 25.0, "scale": 1.0,
                     "source": "command_limits_degrees", "fallback": "0"}],
        "observations": [{"name": "pel", "channels": ["pel_z"], "dim": 1}],
        "reward": [{"label": "alive", "expression": "1", "weight": 0.2}],
        "termination": [{"label": "tipped", "expression": "q", "above": 0.15}],
        "model": {
            "path": "outputs/model-model.xml",
            "sha256": model_sha or hashlib.sha256(MODEL_XML).hexdigest(),
            "bytes": len(MODEL_XML),
        },
    }


@pytest.fixture
def scene(tmp_path: Path, monkeypatch) -> dict:
    """A run directory, a task directory, and ``replay/`` pointed at tmp_path.

    ``SETS`` and ``LEDGER`` are module constants, so they are redirected rather
    than parameterised: a test that wrote into the real ``replay/`` would
    silently enter fabricated sets in the ledger the driver prints on every
    run, which is exactly the obligation the ledger exists to keep honest.
    """

    run = tmp_path / "jobs" / "stand13-20260805-135926"
    run.mkdir(parents=True)
    for tag in (1700, 1750, 1800):
        (run / f"stand13.{tag:06d}.cxpolicy").write_bytes(
            b"CXPOLICY" + str(tag).encode()
        )
    # A `best` file, which carries no periodic tag and would sort between
    # 001750 and 001800 by name.
    (run / "stand13.best.cxpolicy").write_bytes(b"CXPOLICYbest")
    (run / "hyperparameters.json").write_text(json.dumps({"seed": 2}))
    (run / "model-model.xml").write_bytes(MODEL_XML)

    tasks = tmp_path / "tasks" / "clamp25"
    tasks.mkdir(parents=True)
    (tasks / "stand-task.json").write_text(json.dumps(_bundle(), indent=2))
    (tasks / "model-model.xml").write_bytes(MODEL_XML)

    script = tmp_path / "script-clamp25.py"
    script.write_text("# a script\n")

    monkeypatch.setattr(replay, "SETS", tmp_path / "replay")
    monkeypatch.setattr(replay, "LEDGER", tmp_path / "replay" / "ledger.json")
    # ``manifest_for`` records the script relative to the repo root, so the
    # fabricated script has to look like it lives under one.
    monkeypatch.setattr(replay, "REPO_ROOT", tmp_path)
    return {"run": run, "task": tasks / "stand-task.json",
            "model": tasks / "model-model.xml", "script": script,
            "root": tmp_path}


def _export(scene: dict, **kwargs) -> tuple[Path, dict]:
    options = {
        "run_dir": scene["run"], "iteration": 1800, "arm": "clamp25",
        "script": scene["script"], "task": scene["task"], "policy": None,
        "label": None, "destination": None,
    }
    options.update(kwargs)
    return replay.export_set(**options)


# ---------------------------------------------------------------------------
# Choosing the checkpoint
# ---------------------------------------------------------------------------


def test_the_iteration_convention_matches_capture(scene) -> None:
    """``--iteration 1800`` means the file tagged 001800, on both drivers.

    There are two conventions on disk and they differ by one:
    ``series_checkpoints`` reads the filename tag, ``discover_policies`` reads
    the trainer's index out of ``progress.json``. The first version of this
    driver used the second and refused ``--iteration 1800`` on a run holding
    exactly that file. Two drivers disagreeing about what one flag means is the
    failure this pins shut.
    """

    from harness.capture import select_policies

    chosen = replay.checkpoint_for(scene["run"], 1800)
    assert chosen.name == "stand13.001800.cxpolicy"
    assert select_policies(scene["run"], [], 1800)[0][1] == chosen


def test_no_iteration_takes_the_last_one_written(scene) -> None:
    assert replay.checkpoint_for(scene["run"], None).name == \
        "stand13.001800.cxpolicy"


def test_the_best_file_is_not_mistaken_for_a_checkpoint(scene) -> None:
    """It carries no periodic tag, and sorts between 001750 and 001800."""

    assert replay.checkpoint_for(scene["run"], None).name != \
        "stand13.best.cxpolicy"


def test_an_absent_iteration_lists_what_there_is(scene) -> None:
    with pytest.raises(ReplayError) as raised:
        replay.checkpoint_for(scene["run"], 1234)
    message = str(raised.value)
    assert "1234" in message
    for tag in ("1700", "1750", "1800"):
        assert tag in message


# ---------------------------------------------------------------------------
# Finding the model the bundle actually names
# ---------------------------------------------------------------------------


def test_the_model_is_found_by_digest_not_by_name(scene, tmp_path) -> None:
    decoy = tmp_path / "model-model.xml"
    decoy.write_bytes(b"<mujoco model=/>")
    found = replay.model_beside(_bundle(), [decoy, scene["model"]])
    assert found == scene["model"]


def test_a_model_that_matches_nothing_is_refused_loudly(scene) -> None:
    """Shipping a policy with a model it was not trained against would make
    the far machine's equivalence check compare the wrong two mechanisms."""

    with pytest.raises(ReplayError) as raised:
        replay.model_beside(_bundle("0" * 64), [scene["model"]])
    assert "0000" in str(raised.value)
    assert str(scene["model"]) in str(raised.value)


def test_a_bundle_with_no_model_digest_is_refused(scene) -> None:
    bundle = _bundle()
    bundle["model"].pop("sha256")
    with pytest.raises(ReplayError, match="model.sha256"):
        replay.model_beside(bundle, [scene["model"]])


# ---------------------------------------------------------------------------
# The set, and its manifest
# ---------------------------------------------------------------------------


def test_an_export_stages_three_files_and_a_manifest(scene) -> None:
    set_dir, manifest = _export(scene)
    assert sorted(path.name for path in set_dir.iterdir()) == [
        "clamp25-model.xml", "clamp25-task.json", "manifest.json",
        "stand13.001800.cxpolicy",
    ]
    assert manifest["schema"] == replay.SET_SCHEMA
    assert manifest["arm"] == "clamp25"
    assert manifest["run"]["seed_trained"] == 2
    assert manifest["run"]["iteration"] == 1800


def test_the_bundle_and_model_are_named_for_the_arm_not_the_checkpoint(
    scene,
) -> None:
    """Because the committed script names the bundle **literally**.

    ``assembly.policy(..., trained_task="clamp25-task.json")`` is in a file
    under git. Naming the staged bundle for the checkpoint would break that
    line every time the checkpoint moved, which is the whole reason the two
    are renamed at all.
    """

    _set_dir, first = _export(scene, iteration=1800)
    _set_dir, second = _export(scene, iteration=1700, label="clamp25")
    assert first["task"]["file"] == second["task"]["file"] == "clamp25-task.json"
    assert first["model"]["file"] == second["model"]["file"] == "clamp25-model.xml"
    # ...and the policy keeps the name the run wrote, which is how a
    # checkpoint is identified everywhere else in this repository.
    assert first["policy"]["file"] == "stand13.001800.cxpolicy"
    assert second["policy"]["file"] == "stand13.001700.cxpolicy"


def test_the_script_does_not_travel_but_its_digest_does(scene) -> None:
    """It is committed, so both machines have it. A copy would let a set be
    built against a script that is not the one in git."""

    set_dir, manifest = _export(scene)
    assert not (set_dir / "script-clamp25.py").exists()
    assert manifest["script"] == "script-clamp25.py"
    assert manifest["script_sha256"] == hashlib.sha256(
        scene["script"].read_bytes()
    ).hexdigest()


def test_the_manifest_restates_the_bundles_own_model_digest(scene) -> None:
    """So a reader can check a set is coherent without parsing 30 kB of JSON,
    and because the far machine's engine finds the model by that digest."""

    _set_dir, manifest = _export(scene)
    assert manifest["task_model_sha256"] == manifest["model"]["sha256"]


def test_an_export_without_a_run_needs_an_explicit_policy(scene) -> None:
    with pytest.raises(ReplayError, match="--dir"):
        _export(scene, run_dir=None, policy=None)


# ---------------------------------------------------------------------------
# Verifying one, which is what makes a transfer trustworthy
# ---------------------------------------------------------------------------


def test_a_freshly_exported_set_verifies(scene) -> None:
    set_dir, manifest = _export(scene)
    assert replay.verify_set(set_dir, manifest) == []


def test_every_role_is_checked_and_all_complaints_come_back_at_once(
    scene,
) -> None:
    """"The policy is fine and the model is truncated" is more useful than the
    first half of it."""

    set_dir, manifest = _export(scene)
    (set_dir / manifest["model"]["file"]).write_bytes(b"truncated")
    (set_dir / manifest["task"]["file"]).unlink()
    complaints = replay.verify_set(set_dir, manifest)
    assert len(complaints) == 2
    assert any("model" in line and "digests to" in line for line in complaints)
    assert any("task" in line and "missing" in line for line in complaints)
    # ...and the policy is not complained about.
    assert not any(line.startswith("policy") for line in complaints)


def test_a_byte_count_that_disagrees_is_a_complaint(scene) -> None:
    set_dir, manifest = _export(scene)
    manifest["policy"]["bytes"] = 1
    assert any("bytes" in line for line in replay.verify_set(set_dir, manifest))


def test_an_import_refuses_a_set_that_does_not_match_its_manifest(
    scene, tmp_path
) -> None:
    """A digest checked at the far end is the point. A truncated .cxpolicy
    would otherwise surface as ``decode_policy`` refusing a container three
    steps later, which reads like a corrupt file rather than a bad copy."""

    set_dir, manifest = _export(scene)
    (set_dir / manifest["policy"]["file"]).write_bytes(b"short")
    with pytest.raises(ReplayError, match="does not match its own manifest"):
        replay.import_set(set_dir, tmp_path / "arrived")
    assert not (tmp_path / "arrived").exists()


def test_an_import_copies_and_re_checks_on_arrival(scene, tmp_path) -> None:
    set_dir, _manifest = _export(scene)
    target, manifest = replay.import_set(set_dir, tmp_path / "arrived")
    assert replay.verify_set(target, manifest) == []
    for role in replay.ROLES:
        name = manifest[role]["file"]
        assert (target / name).read_bytes() == (set_dir / name).read_bytes()


def test_a_directory_without_a_manifest_is_not_a_set(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ReplayError, match="not a replay set"):
        replay.read_set(tmp_path / "empty")


def test_a_manifest_of_another_schema_is_refused(scene) -> None:
    set_dir, _manifest = _export(scene)
    path = set_dir / "manifest.json"
    stored = json.loads(path.read_text())
    stored["schema"] = "cdxrl-replay-set-v99"
    path.write_text(json.dumps(stored))
    with pytest.raises(ReplayError, match="v99"):
        replay.read_set(set_dir)


# ---------------------------------------------------------------------------
# Script drift, which is a note rather than a refusal
# ---------------------------------------------------------------------------


def test_a_matching_script_reports_no_drift(scene) -> None:
    _set_dir, manifest = _export(scene)
    assert replay.script_drift(manifest) is None


def test_an_edited_script_is_reported_and_not_refused(scene) -> None:
    """The build that follows is what can actually tell; this is a heads-up."""

    _set_dir, manifest = _export(scene)
    scene["script"].write_text("# edited\n")
    drift = replay.script_drift(manifest)
    assert drift is not None
    assert "script-clamp25.py" in drift


def test_a_script_missing_from_this_checkout_is_reported(scene) -> None:
    _set_dir, manifest = _export(scene)
    scene["script"].unlink()
    assert "not in this checkout" in (replay.script_drift(manifest) or "")


# ---------------------------------------------------------------------------
# Resolving a set by label, by arm, or by path
# ---------------------------------------------------------------------------


def test_a_set_resolves_by_label_and_by_path(scene) -> None:
    set_dir, _manifest = _export(scene, label="stand13.001800")
    assert replay.find_set("stand13.001800") == set_dir      # label
    assert replay.find_set(str(set_dir)) == set_dir           # path


def test_a_set_resolves_by_its_arm_when_only_one_declares_it(scene) -> None:
    set_dir, _manifest = _export(scene, label="stand13.001800")
    assert replay.find_set("clamp25") == set_dir


def test_two_sets_of_one_arm_refuse_to_be_guessed_between(scene) -> None:
    """Both name the arm, so the arm is not an answer. It says which two."""

    first, _ = _export(scene, label="stand13.001800")
    second, _ = _export(scene, label="stand13.001700", iteration=1700)
    assert first != second
    with pytest.raises(ReplayError) as raised:
        replay.find_set("clamp25")
    message = str(raised.value)
    assert "2 sets declare arm" in message
    assert "stand13.001800" in message and "stand13.001700" in message
    # ...and each label still resolves on its own.
    assert replay.find_set("stand13.001700") == second


def test_an_unknown_name_says_how_to_look(scene) -> None:
    with pytest.raises(ReplayError, match="--list"):
        replay.find_set("nothing-like-this")


# ---------------------------------------------------------------------------
# The ledger: every set owes a Flywheel node
# ---------------------------------------------------------------------------


def test_an_export_is_recorded_as_owing_a_node(scene) -> None:
    set_dir, manifest = _export(scene)
    items = replay.upload_items(set_dir, manifest)
    replay.record(set_dir, manifest, items)
    outstanding = replay.pending()
    assert len(outstanding) == 1
    assert outstanding[0]["set_dir"] == str(set_dir)


def test_four_items_go_up_and_a_table_is_json(scene) -> None:
    """A ``table`` artifact on Flywheel must be JSON, and a ``.cxpolicy``
    uploads as ``binary`` (flywheel.md §5). The MJCF is XML, so it is binary
    too."""

    set_dir, manifest = _export(scene)
    items = replay.upload_items(set_dir, manifest)
    by_role = {item["metadata"]["role"]: item for item in items}
    assert sorted(by_role) == ["manifest", "model", "policy", "task"]
    assert by_role["task"]["artifact_type"] == "table"
    assert by_role["manifest"]["artifact_type"] == "table"
    assert by_role["policy"]["artifact_type"] == "binary"
    assert by_role["model"]["artifact_type"] == "binary"
    for role in ("policy", "task", "model"):
        assert by_role[role]["metadata"]["sha256"] == manifest[role]["sha256"]
        assert by_role[role]["metadata"]["bytes"] == manifest[role]["bytes"]


def test_metadata_never_says_a_bare_seed(scene) -> None:
    """One seed decides which policy you got; another decides which scenario
    you played it on (flywheel-conventions.md §3)."""

    set_dir, manifest = _export(scene)
    for item in replay.upload_items(set_dir, manifest):
        assert "seed" not in item["metadata"]
        assert item["metadata"]["seed_trained"] == 2


def test_publishing_then_re_exporting_the_same_bytes_keeps_the_node(
    scene,
) -> None:
    set_dir, manifest = _export(scene)
    replay.record(set_dir, manifest, replay.upload_items(set_dir, manifest))
    replay.mark_published("sparkling-cherry-4343")
    assert replay.pending() == []

    set_dir, manifest = _export(scene)
    replay.record(set_dir, manifest, replay.upload_items(set_dir, manifest))
    assert replay.pending() == []


def test_a_re_export_with_new_bytes_owes_a_new_artifact(scene) -> None:
    """Keeping the node id would claim the graph holds these bytes when it
    holds different ones. Same rule ``capture``'s ledger keeps."""

    set_dir, manifest = _export(scene, iteration=1800)
    replay.record(set_dir, manifest, replay.upload_items(set_dir, manifest))
    replay.mark_published("sparkling-cherry-4343")
    assert replay.pending() == []

    set_dir, manifest = _export(scene, iteration=1700, label=manifest["label"])
    replay.record(set_dir, manifest, replay.upload_items(set_dir, manifest))
    assert len(replay.pending()) == 1


def test_a_set_deleted_before_it_reached_the_graph_is_not_pending(
    scene,
) -> None:
    """There are no bytes to upload, so it is not an obligation. It is still
    *reported* by ``--pending``, because a ledger that quietly forgot it would
    make "every set is on the graph" true by attrition."""

    set_dir, manifest = _export(scene)
    replay.record(set_dir, manifest, replay.upload_items(set_dir, manifest))
    shutil.rmtree(set_dir)
    assert replay.pending() == []
    ledger = replay.read_ledger()
    assert manifest["label"] in ledger["sets"]


def test_the_note_precedes_what_the_driver_can_know(scene) -> None:
    set_dir, manifest = _export(scene)
    items = replay.upload_items(set_dir, manifest, note="why this checkpoint")
    for item in items:
        assert item["note"].startswith("why this checkpoint")
        assert "trained_task" in item["note"]


# ---------------------------------------------------------------------------
# The transport with no record
# ---------------------------------------------------------------------------


def test_the_scp_line_names_the_set_and_the_import(scene) -> None:
    set_dir, _manifest = _export(scene)
    line = replay.scp_line(set_dir, "mmini")
    assert str(set_dir) in line
    assert "mmini:~/cdx-rl/replay/" in line
    assert "--import" in line
