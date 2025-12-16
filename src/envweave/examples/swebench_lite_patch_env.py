from __future__ import annotations

import json
import platform
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from envweave.types import ResetResult, StepResult


@dataclass(frozen=True)
class SWEbenchLitePatchObs:
    prompt: str
    instance_id: str
    repo: str
    base_commit: str
    t: int


@dataclass(frozen=True)
class SWEbenchLitePatchAction:
    patch: str


@dataclass(frozen=True)
class SWEbenchEvalResult:
    ok: bool
    passed: int
    failed: int
    total: int
    image: str
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class SWEbenchRunner(Protocol):
    def evaluate(
        self,
        *,
        instance_id: str,
        test_patch: str,
        model_patch: str,
        tests: list[str],
        timeout_s: float,
    ) -> SWEbenchEvalResult: ...


_PYTEST_COUNTS_RE = re.compile(
    r"(?:(?P<passed>\\d+)\\s+passed)?(?:,\\s*)?"
    r"(?:(?P<failed>\\d+)\\s+failed)?(?:,\\s*)?"
    r"(?:(?P<errors>\\d+)\\s+error[s]?)?",
    re.IGNORECASE,
)


def _default_cache_dir() -> Path:
    return (Path.home() / ".cache" / "envweave" / "swebench_lite").expanduser()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_swebench_lite(split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Loading SWE-bench Lite from Hugging Face requires optional dependency 'datasets'. "
            "Install with: pip install -e '.[swebench]'"
        ) from e

    split = str(split).strip()
    for name in ("SWE-bench/SWE-bench_Lite", "princeton-nlp/SWE-bench_Lite"):
        try:
            ds = load_dataset(name, split=split, streaming=False)  # type: ignore[arg-type]
            return [dict(r) for r in ds]  # type: ignore[return-value]
        except Exception:
            continue
    raise RuntimeError(f"failed to load SWE-bench Lite split={split!r} from Hugging Face")


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except Exception:
                return []
        return [s]
    return [str(value)]


def _docker_arch() -> str:
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.Architecture}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            arch = (proc.stdout or "").strip().lower()
        else:
            arch = ""
    except Exception:
        arch = ""

    if not arch:
        arch = platform.machine().lower()

    if arch in ("aarch64", "arm64"):
        return "arm64"
    if arch in ("x86_64", "amd64"):
        return "x86_64"
    return arch


class DockerSWEbenchRunner:
    """
    Evaluates a candidate patch by running an official SWE-bench instance image.

    Uses GHCR images (Epoch AI registry):
      ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest

    Procedure:
      1) Start container
      2) Apply `test_patch`
      3) Apply `model_patch`
      4) Run pytest for selected tests
    """

    def __init__(
        self,
        *,
        image_template: str = "ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest",
        arch: str | None = None,
        auto_pull: bool = True,
        extra_docker_run_args: list[str] | None = None,
    ) -> None:
        self._image_template = str(image_template)
        self._arch = str(arch).strip() if arch is not None else ""
        self._auto_pull = bool(auto_pull)
        self._extra_run_args = list(extra_docker_run_args or [])

    def _image_for(self, instance_id: str) -> str:
        arch = self._arch or _docker_arch()
        return self._image_template.format(arch=arch, instance_id=str(instance_id))

    @staticmethod
    def _image_exists(image: str) -> bool:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0

    def _ensure_image(self, image: str) -> None:
        if self._image_exists(image):
            return
        if not self._auto_pull:
            raise RuntimeError(f"docker image not found locally: {image}")
        subprocess.run(["docker", "pull", image], check=True)

    def evaluate(
        self,
        *,
        instance_id: str,
        test_patch: str,
        model_patch: str,
        tests: list[str],
        timeout_s: float,
    ) -> SWEbenchEvalResult:
        t0 = time.perf_counter()
        image = self._image_for(instance_id)
        self._ensure_image(image)

        if not tests:
            return SWEbenchEvalResult(
                ok=False,
                passed=0,
                failed=0,
                total=0,
                image=image,
                duration_s=0.0,
                error="no tests provided",
            )

        with tempfile.TemporaryDirectory(prefix="envweave_swebench_") as td:
            td_path = Path(td)
            (td_path / "test_patch.diff").write_text(str(test_patch or "") + "\n", encoding="utf-8")
            (td_path / "model_patch.diff").write_text(str(model_patch or "") + "\n", encoding="utf-8")
            (td_path / "tests.json").write_text(json.dumps(list(tests)), encoding="utf-8")

            (td_path / "run_pytest.py").write_text(
                (
                    "import json, subprocess, sys\n"
                    "tests = json.load(open('/patches/tests.json','r',encoding='utf-8'))\n"
                    "cmd = ['pytest','-q', *tests]\n"
                    "p = subprocess.run(cmd, cwd='/testbed', text=True, capture_output=True)\n"
                    "sys.stdout.write(p.stdout or '')\n"
                    "sys.stderr.write(p.stderr or '')\n"
                    "raise SystemExit(p.returncode)\n"
                ),
                encoding="utf-8",
            )

            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{td_path}:/patches:ro",
                *self._extra_run_args,
                image,
                "bash",
                "-lc",
                (
                    "set -euo pipefail; "
                    "cd /testbed; "
                    "git reset --hard >/dev/null 2>&1 || true; "
                    "git clean -fdx >/dev/null 2>&1 || true; "
                    "if [ -s /patches/test_patch.diff ]; then git apply /patches/test_patch.diff; fi; "
                    "if [ -s /patches/model_patch.diff ]; then git apply /patches/model_patch.diff; fi; "
                    "python /patches/run_pytest.py"
                ),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_s),
                )
            except subprocess.TimeoutExpired as e:
                duration = time.perf_counter() - t0
                return SWEbenchEvalResult(
                    ok=False,
                    passed=0,
                    failed=len(tests),
                    total=len(tests),
                    image=image,
                    duration_s=float(duration),
                    stdout=(e.stdout or ""),
                    stderr=(e.stderr or ""),
                    error=f"timeout after {timeout_s}s",
                )

        duration = time.perf_counter() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        ok = proc.returncode == 0

        passed = 0
        failed = 0
        m = _PYTEST_COUNTS_RE.search(stdout + "\n" + stderr)
        if m:
            if m.group("passed"):
                passed = int(m.group("passed"))
            if m.group("failed"):
                failed = int(m.group("failed"))
            if m.group("errors"):
                failed += int(m.group("errors"))

        total = max(len(tests), passed + failed, 0)
        if ok:
            passed = total
            failed = 0
        else:
            if passed == 0 and failed == 0:
                failed = total

        err: str | None = None
        if not ok:
            # Keep it short but informative.
            tail = (stderr or stdout).strip().splitlines()[-8:]
            err = "pytest_failed: " + (" | ".join(tail)[:800] if tail else "unknown")

        return SWEbenchEvalResult(
            ok=bool(ok),
            passed=int(passed),
            failed=int(failed),
            total=int(total),
            image=image,
            duration_s=float(duration),
            stdout=stdout[-8000:],
            stderr=stderr[-8000:],
            error=err,
        )


class SWEbenchLitePatchEnv:
    """
    SWE-bench Lite as a patch-generating environment.

    - reset() returns a prompt containing the issue text and a strict patch format request.
    - step(patch_text) evaluates in Docker (official instance image) and returns reward.

    This is designed for end-to-end "generate patch -> run tests" RL loops.
    """

    observation_type = SWEbenchLitePatchObs
    action_type = SWEbenchLitePatchAction

    def __init__(
        self,
        *,
        split: str = "dev",
        seed: int | None = None,
        dataset_path: str | None = None,
        cache_dir: str | None = None,
        max_examples: int | None = None,
        instance_ids: Iterable[str] | None = None,
        max_fail_tests: int = 12,
        timeout_s: float = 600.0,
        runner: SWEbenchRunner | None = None,
    ):
        self._split = str(split).strip()
        self._rng = random.Random(seed)
        self._t = 0
        self._timeout_s = float(timeout_s)
        self._max_fail_tests = int(max_fail_tests)
        self._runner: SWEbenchRunner = runner or DockerSWEbenchRunner()

        if dataset_path:
            rows = _load_jsonl(Path(dataset_path).expanduser())
        else:
            cache = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            rows = _load_swebench_lite(self._split)

        if instance_ids is not None:
            wanted = {str(x) for x in instance_ids}
            rows = [r for r in rows if str(r.get("instance_id", "")) in wanted]
        if max_examples is not None:
            rows = rows[: int(max_examples)]
        if not rows:
            raise ValueError("no SWE-bench Lite rows loaded")

        normalized: list[dict[str, Any]] = []
        for r in rows:
            instance_id = str(r.get("instance_id", "") or "").strip()
            repo = str(r.get("repo", "") or "").strip()
            base_commit = str(r.get("base_commit", "") or "").strip()
            problem = str(r.get("problem_statement", "") or "").strip()
            test_patch = str(r.get("test_patch", "") or "")
            if not instance_id or not repo or not base_commit or not problem:
                continue
            normalized.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "problem_statement": problem,
                    "patch": str(r.get("patch", "") or ""),
                    "test_patch": test_patch,
                    "fail_to_pass": _json_list(r.get("FAIL_TO_PASS")),
                    "pass_to_pass": _json_list(r.get("PASS_TO_PASS")),
                }
            )
        if not normalized:
            raise ValueError("no usable SWE-bench Lite rows after normalization")
        self._rows = normalized

        self._row: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng.seed(int(seed))
        self._t = 0

        row = self._rng.choice(self._rows)
        self._row = row

        prompt = (
            "You are solving a SWE-bench Lite GitHub issue.\n"
            "Return ONLY a unified diff patch (no markdown code fences, no explanation).\n"
            "The patch MUST apply cleanly with `git apply`.\n\n"
            f"Repo: {row['repo']}\n"
            f"Instance: {row['instance_id']}\n"
            f"Base commit: {row['base_commit']}\n\n"
            "Issue:\n"
            f"{row['problem_statement']}\n\n"
            "Patch:\n"
        )
        return ResetResult(
            obs=SWEbenchLitePatchObs(
                prompt=prompt,
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                t=self._t,
            ),
            info={"instance_id": row["instance_id"], "repo": row["repo"], "split": self._split},
        )

    def step(self, action: SWEbenchLitePatchAction | dict[str, Any] | str):
        if self._row is None:
            raise RuntimeError("step() called before reset()")

        patch_text = self._parse_patch(action)
        self._t += 1

        tests = list(self._row.get("fail_to_pass") or [])
        if self._max_fail_tests > 0:
            tests = tests[: int(self._max_fail_tests)]

        result = self._runner.evaluate(
            instance_id=str(self._row["instance_id"]),
            test_patch=str(self._row.get("test_patch", "") or ""),
            model_patch=str(patch_text),
            tests=[str(t) for t in tests],
            timeout_s=float(self._timeout_s),
        )

        reward = 1.0 if bool(result.ok) else -1.0
        info = {
            "terminated": True,
            "truncated": False,
            "correct": bool(result.ok),
            "instance_id": str(self._row["instance_id"]),
            "repo": str(self._row["repo"]),
            "base_commit": str(self._row["base_commit"]),
            "split": self._split,
            "tests_run": int(result.total),
            "passed": int(result.passed),
            "failed": int(result.failed),
            "duration_s": float(result.duration_s),
            "image": str(result.image),
            "error": result.error,
        }
        return StepResult(
            obs=SWEbenchLitePatchObs(
                prompt="",
                instance_id=str(self._row["instance_id"]),
                repo=str(self._row["repo"]),
                base_commit=str(self._row["base_commit"]),
                t=self._t,
            ),
            reward=float(reward),
            done=True,
            info=info,
            requests=[],
        )

    @staticmethod
    def _parse_patch(action: SWEbenchLitePatchAction | dict[str, Any] | str) -> str:
        if isinstance(action, SWEbenchLitePatchAction):
            return str(action.patch)
        if isinstance(action, dict):
            return str(action.get("patch", ""))
        if isinstance(action, str):
            return action
        raise TypeError(f"unsupported action type: {type(action).__name__}")

    def close(self) -> None:
        return None


__all__ = [
    "SWEbenchLitePatchEnv",
    "SWEbenchLitePatchObs",
    "SWEbenchLitePatchAction",
    "DockerSWEbenchRunner",
    "SWEbenchEvalResult",
]
