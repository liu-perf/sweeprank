"""Every `run:` block in ``ci.yml``, executed.

Three repositories in this series shipped a CI that was red on every push
while ``pytest``, ``ruff`` and every other check stayed green -- because the
defect was in the YAML and every test was in Python. Two of them had a CLI
option after the subcommand, so argparse exited 2, which no
``|| test $? -eq 1`` can absorb; one was a missing ``!`` on a line that was
supposed to expect failure.

So this test runs the shell. It skips the steps that need the network or the
installed console script and runs the rest with ``bash -e``, substituting
``python -m sweeprank.cli`` for the bare ``sweeprank`` entry point so the suite
does not depend on ``pip install -e .`` having happened.

It is skipped rather than failed where bash is unavailable, because a Windows
box without bash is a fact about the box and not about the workflow.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

CI = pathlib.Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
REPO = CI.parent.parent.parent
BASH = shutil.which("bash")

# Steps that need something CI has and this test does not.
NEEDS_NETWORK = ("pip install", "actions/")

# `- run: pytest -q` is the step that runs this file. Executing it here would
# recurse until the machine gave up, which is what the first version of this
# test did. It is skipped by name and asserted to exist instead, so the skip
# cannot quietly become a workflow with no test step at all.
SELF_REFERENTIAL = ("pytest",)


def _steps():
    """(name, script) for every `run:` in the workflow.

    Deliberately a small hand parser rather than a YAML dependency: this
    package has none, and adding one for a test would make the CI it checks
    depend on it too.
    """
    lines = CI.read_text(encoding="utf-8").split("\n")
    out, name = [], None
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^\s*- name:\s*(.+)$", ln)
        if m:
            name = m.group(1).strip()
            i += 1
            continue
        m = re.match(r"^(\s*)-?\s*run:\s*(.*)$", ln)
        if m:
            first = m.group(2).strip()
            if first in ("|", ">"):
                body, i = [], i + 1
                block_indent = None
                while i < len(lines):
                    nxt = lines[i]
                    if nxt.strip() == "":
                        body.append("")
                        i += 1
                        continue
                    lead = len(nxt) - len(nxt.lstrip())
                    if block_indent is None:
                        block_indent = lead
                    if lead < block_indent:
                        break
                    body.append(nxt[block_indent:])
                    i += 1
                out.append((name or "<unnamed>", "\n".join(body)))
            else:
                out.append((name or "<unnamed>", first))
                i += 1
            name = None
            continue
        i += 1
    return out


def _runnable():
    out = []
    for n, s in _steps():
        if not s.strip():
            continue
        if any(k in s for k in NEEDS_NETWORK):
            continue
        if s.strip().split()[0] in SELF_REFERENTIAL:
            continue
        out.append((n, s))
    return out


def test_the_workflow_still_runs_the_test_suite():
    """The step this test refuses to execute has to still be there."""
    assert any(s.strip().split()[:1] == ["pytest"] for _, s in _steps()), (
        "ci.yml no longer runs pytest; test_ci_runs skips that step by name and "
        "would happily pass on a workflow that never tests anything"
    )


def test_the_parser_found_the_steps():
    steps = _steps()
    assert len(steps) >= 12, [n for n, _ in steps]
    names = [n for n, _ in steps]
    assert "the report is still loud" in names
    assert any("proxy still cannot select" in n for n in names)


def test_at_least_one_step_asserts_a_failure_path():
    """The missing-`!` bug, as a test.

    A workflow whose every line expects success cannot notice that a check
    stopped having teeth. At least one step has to be checking that something
    fails.
    """
    blocks = "\n".join(s for _, s in _steps())
    assert "exit 1" in blocks or re.search(r"^\s*!\s", blocks, re.M)


@pytest.fixture(scope="module")
def shim_path(tmp_path_factory):
    """A directory on PATH holding a ``sweeprank`` that runs this source tree.

    The first version of this test rewrote the workflow text with a regex
    instead, and ``\\bsweeprank\\b`` promptly matched ``sweeprank.model`` inside
    the heredocs -- so nine steps failed with a Python ``SyntaxError`` that had
    nothing to do with the workflow. A checker whose failures are its own is
    worse than no checker, because people start ignoring it. Shims mean the
    YAML runs **verbatim**.
    """
    d = tmp_path_factory.mktemp("bin")
    exe = sys.executable.replace("\\", "/")
    for name, args in (("sweeprank", "-X utf8 -m sweeprank.cli"),
                       ("ruff", "-m ruff"),
                       ("python", ""), ("python3", "")):
        p = d / name
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f'#!/bin/sh\nexec "{exe}" {args} "$@"\n')
        p.chmod(0o755)
    return str(d).replace("\\", "/")


@pytest.mark.skipif(BASH is None, reason="no bash on this machine")
@pytest.mark.parametrize("name,script", _runnable(), ids=[n for n, _ in _runnable()])
def test_ci_step_runs(name, script, shim_path, monkeypatch):
    env = dict(os.environ)
    env["PATH"] = shim_path + os.pathsep.replace(";", ":") + env.get("PATH", "")
    proc = subprocess.run([BASH, "-e", "-c", script], cwd=REPO, check=False,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    assert proc.returncode == 0, (
        f"CI step {name!r} exits {proc.returncode}\n"
        f"--- script ---\n{script}\n--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )
