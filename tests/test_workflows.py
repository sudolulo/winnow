"""Guards for the release / docker-publish workflow split.

The image jobs in release.yml are deliberate copies of the jobs in
docker-publish.yml, gated on the release job. These tests keep the copies
aligned, pin the triggers, and keep step outputs and other expression values
out of shell source. PyYAML is not a project dependency, so the workflows are
read with a small indentation-based parser that only understands the subset of
YAML these files use.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
DOCKER = WORKFLOWS / "docker-publish.yml"
IMAGE_JOBS = ["build", "merge", "build-cpu", "build-rocm", "build-intel"]
RELEASE_NEEDS = {
    "build": ["release"],
    "merge": ["release", "build"],
    "build-cpu": ["release"],
    "build-rocm": ["release"],
    "build-intel": ["release"],
}
REGISTRY = "ghcr.io"
IMAGE_NAME = "sudolulo/winnow"
IMAGE = f"{REGISTRY}/{IMAGE_NAME}"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _lines(path):
    return path.read_text().splitlines()


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _strip_comment(line):
    # Good enough for these files: no '#' appears inside quoted values that
    # matter to the checks below.
    return re.sub(r"\s+#.*$", "", line) if not line.lstrip().startswith("#") else ""


def top_block(path, key):
    """Lines (comments dropped) under a column-0 ``key:``."""
    out, inside = [], False
    for line in _lines(path):
        if re.match(rf"^{re.escape(key)}:\s*$", line):
            inside = True
            continue
        if inside and line and not line[0].isspace() and not line.startswith("#"):
            break
        if inside:
            stripped = _strip_comment(line)
            if stripped.strip():
                out.append(stripped)
    return out


def jobs(path):
    """Map job id -> raw body lines (including comments)."""
    result, current, inside = {}, None, False
    for line in _lines(path):
        if line == "jobs:":
            inside = True
            continue
        if not inside:
            continue
        if line and not line[0].isspace():
            break
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if m:
            current = m.group(1)
            result[current] = []
        elif current is not None:
            result[current].append(line)
    for body in result.values():
        while body and not body[-1].strip():
            body.pop()
    return result


def job_needs(body):
    for line in body:
        m = re.match(r"^    needs:\s*(.+?)\s*$", line)
        if m:
            value = m.group(1)
            if value.startswith("["):
                return [v.strip() for v in value.strip("[]").split(",") if v.strip()]
            return [value]
    return None


def without_needs(body):
    return [line for line in body if not re.match(r"^    needs:", line)]


def steps(body):
    """Parse ``steps:`` of a job body into dicts with name/id/run/env."""
    result, step, mode = [], None, None
    for line in body:
        if re.match(r"^      - ", line):
            step = {"run": None, "env": {}}
            result.append(step)
            line = "        " + line[8:]
            mode = None
        if step is None:
            continue
        if mode == "run":
            if not line.strip() or _indent(line) >= 10:
                step["run"].append(line[10:])
                continue
            mode = None
        if mode == "env":
            m = re.match(r"^          ([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
            if m:
                step["env"][m.group(1)] = m.group(2).strip().strip('"')
                continue
            mode = None
        m = re.match(r"^        (name|id|run|env|working-directory):\s*(.*)$", line)
        if not m:
            continue
        key, value = m.groups()
        if key == "run":
            if value in ("|", "|-"):
                step["run"] = []
                mode = "run"
            else:
                step["run"] = [value]
        elif key == "env":
            mode = "env"
        else:
            step[key] = value.strip().strip('"')
    for s in result:
        if s["run"] is not None:
            s["run"] = "\n".join(s["run"]).rstrip() + "\n"
    return result


def step_by(path, job, **match):
    for s in steps(jobs(path)[job]):
        if all(s.get(k) == v for k, v in match.items()):
            return s
    raise AssertionError(f"{path.name}:{job} has no step {match}")


# --- structure ------------------------------------------------------------


def test_release_has_release_root_and_all_image_jobs():
    rel = jobs(RELEASE)
    assert set(rel) == {"release", *IMAGE_JOBS}
    assert job_needs(rel["release"]) is None


@pytest.mark.parametrize("job", IMAGE_JOBS)
def test_release_image_jobs_need_release(job):
    assert job_needs(jobs(RELEASE)[job]) == RELEASE_NEEDS[job]


@pytest.mark.parametrize("job", IMAGE_JOBS)
def test_image_job_copies_aligned_ignoring_needs(job):
    assert without_needs(jobs(RELEASE)[job]) == without_needs(jobs(DOCKER)[job])


def test_docker_publish_keeps_its_own_needs():
    dp = jobs(DOCKER)
    assert set(dp) == set(IMAGE_JOBS)
    assert job_needs(dp["merge"]) == ["build"]
    for job in ("build", "build-cpu", "build-rocm", "build-intel"):
        assert job_needs(dp[job]) is None


# --- triggers -------------------------------------------------------------


def test_release_triggers_on_v_tags_only():
    assert [line.strip() for line in top_block(RELEASE, "on")] == ["push:", 'tags: ["v*"]']


def test_docker_publish_triggers_on_dev_branch_only():
    on = [line.strip() for line in top_block(DOCKER, "on")]
    assert on[0] == "push:"
    assert 'branches: ["dev"]' in on
    assert not any(line.startswith("tags:") for line in on)
    assert not any(line.startswith(("pull_request", "schedule")) for line in on)


@pytest.mark.parametrize("path", [RELEASE, DOCKER], ids=lambda p: p.name)
def test_no_reusable_or_manual_entry_points(path):
    code = "\n".join(_strip_comment(line) for line in _lines(path))
    for word in ("workflow_call", "workflow_dispatch", "inputs."):
        assert word not in code


# --- expression injection guard -------------------------------------------


@pytest.mark.parametrize("path", [RELEASE, DOCKER], ids=lambda p: p.name)
def test_run_scripts_take_no_expression_values_except_static_env(path):
    offenders = []
    for job, body in jobs(path).items():
        for s in steps(body):
            if s["run"] is None:
                continue
            for expr in re.findall(r"\$\{\{(.*?)\}\}", s["run"]):
                if not re.fullmatch(r"\s*env\.(REGISTRY|IMAGE_NAME)\s*", expr):
                    offenders.append(f"{job}/{s.get('name')}: ${{{{{expr}}}}}")
    assert offenders == []


# --- shell behaviour --------------------------------------------------------


def _render(script):
    return script.replace("${{ env.REGISTRY }}", REGISTRY).replace("${{ env.IMAGE_NAME }}", IMAGE_NAME)


def _run(script, tmp_path, env=None, cwd=None):
    out = tmp_path / "github_output"
    out.write_text("")
    full_env = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "GITHUB_OUTPUT": str(out),
        "REGISTRY": REGISTRY,
        "IMAGE_NAME": IMAGE_NAME,
        **(env or {}),
    }
    proc = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _render(script)],
        env=full_env,
        cwd=cwd or tmp_path,
        capture_output=True,
        text=True,
    )
    return proc, _parse_output(out.read_text())


def _parse_output(text):
    result, lines, i = {}, text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([A-Za-z_]+)<<(\S+)$", line)
        if m:
            key, marker, values = m.group(1), m.group(2), []
            i += 1
            while lines[i] != marker:
                values.append(lines[i])
                i += 1
            result[key] = "\n".join(values)
        elif "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
        i += 1
    return result


@needs_bash
@pytest.mark.parametrize(
    ("tag", "version"),
    [("v0.6.6", "0.6.6"), ("v10.20.30", "10.20.30"), ("v1.0.0-rc.1", "1.0.0-rc.1")],
)
def test_release_tag_resolution_accepts_versions(tag, version, tmp_path):
    s = step_by(RELEASE, "release", id="tag")
    proc, out = _run(s["run"], tmp_path, {"GITHUB_REF_NAME": tag})
    assert proc.returncode == 0, proc.stderr
    assert out == {"TAG": tag, "VERSION": version}


@needs_bash
@pytest.mark.parametrize(
    "tag",
    ["v1.2", "vfoo", "1.2.3", "v1.2.3;id", "v1.2.3$(id)", "v1.2.3\nX=1", "v1.2.3+build", "v1.2.3-", ""],
)
def test_release_tag_resolution_rejects_unexpected_tags(tag, tmp_path):
    s = step_by(RELEASE, "release", id="tag")
    proc, out = _run(s["run"], tmp_path, {"GITHUB_REF_NAME": tag})
    assert proc.returncode != 0
    assert out == {}


@needs_bash
def test_changelog_extraction_uses_version_from_env(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-01-01\n\n- one\n\n## [1.2.2] - 2025-01-01\n\n- old\n"
    )
    s = step_by(RELEASE, "release", id="changelog")
    assert s["env"].get("VERSION") == "${{ steps.tag.outputs.VERSION }}"
    proc, out = _run(s["run"], tmp_path, {"VERSION": "1.2.3"})
    assert proc.returncode == 0, proc.stderr
    assert out["NOTES"].strip() == "- one"
    # A dot in the version must not act as a regex wildcard.
    proc, out = _run(s["run"], tmp_path, {"VERSION": "1x2x3"})
    assert proc.returncode == 0, proc.stderr
    assert out["NOTES"] == ""


def _fake_docker(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "docker_args"
    fake = bindir / "docker"
    fake.write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > "{log}"\n')
    fake.chmod(0o755)
    return log


def _step_env(s, outputs):
    env = {}
    for key, value in s["env"].items():
        m = re.fullmatch(r"\$\{\{\s*steps\.[A-Za-z0-9_-]+\.outputs\.([A-Za-z_]+)\s*\}\}", value)
        assert m, f"unexpected env value {value!r}"
        env[key] = outputs[m.group(1)]
    return env


@needs_bash
@pytest.mark.parametrize("path", [RELEASE, DOCKER], ids=lambda p: p.name)
@pytest.mark.parametrize(
    ("ref", "want_tags", "want_inspect"),
    [
        ("refs/tags/v1.2.3", [f"{IMAGE}:latest", f"{IMAGE}:v1.2.3"], f"{IMAGE}:latest"),
        ("refs/heads/dev", [f"{IMAGE}:dev"], f"{IMAGE}:dev"),
    ],
)
def test_merge_passes_tags_as_separate_arguments(path, ref, want_tags, want_inspect, tmp_path):
    log = _fake_docker(tmp_path)
    tags_step = step_by(path, "merge", id="tags")
    proc, outputs = _run(tags_step["run"], tmp_path, {"GITHUB_REF": ref})
    assert proc.returncode == 0, proc.stderr
    assert outputs["inspect_tag"] == want_inspect

    digests = tmp_path / "digests"
    digests.mkdir()
    (digests / "aaa").write_text("")
    (digests / "bbb").write_text("")
    create = step_by(path, "merge", name="Create and push multi-arch manifest")
    proc, _ = _run(create["run"], tmp_path, _step_env(create, outputs), cwd=digests)
    assert proc.returncode == 0, proc.stderr
    want = ["buildx", "imagetools", "create"]
    for tag in want_tags:
        want += ["-t", tag]
    want += [f"{IMAGE}@sha256:aaa", f"{IMAGE}@sha256:bbb"]
    assert log.read_text().splitlines() == want

    inspect = step_by(path, "merge", name="Inspect image")
    proc, _ = _run(inspect["run"], tmp_path, _step_env(inspect, outputs))
    assert proc.returncode == 0, proc.stderr
    assert log.read_text().splitlines() == ["buildx", "imagetools", "inspect", want_inspect]


def test_workflows_live_where_gitea_reads_them():
    # Gitea reads only the FIRST of .gitea/workflows and .github/workflows that exists;
    # a .gitea/workflows directory would silently stop every workflow in .github.
    assert not (ROOT / ".gitea" / "workflows").exists()
    assert (WORKFLOWS / "secret-scan.yml").is_file()
