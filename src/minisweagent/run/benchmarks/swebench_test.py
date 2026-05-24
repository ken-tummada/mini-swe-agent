#!/usr/bin/env python3

"""Apply patches and run SWE-bench test suites without any agent/LLM runs."""

import base64
import concurrent.futures
import json
import re
import threading
import time
import traceback
from pathlib import Path

import typer
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    filter_instances,
    get_sb_environment,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

NON_TEST_EXTS = frozenset(
    {
        ".pyc",
        ".txt",
        ".rst",
        ".md",
        ".cfg",
        ".ini",
        ".yml",
        ".yaml",
        ".json",
        ".xml",
        ".css",
        ".js",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".toml",
    }
)

from minisweagent.run.benchmarks._swebench_specs import MAP_REPO_VERSION_TO_SPECS_PY

# All SWE-bench eval images install the test runner into a conda env named
# `testbed` at /opt/miniconda3. The default shell (`bash -c`) is non-login and
# does not auto-activate it, so binaries like pytest/tox/django aren't on PATH.
# Prepend this to every test command so the runner is actually found.
CONDA_ACTIVATE = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


def get_test_directives(test_patch: str) -> list[str]:
    directives = re.findall(r"diff --git a/.*? b/(.*)", test_patch)
    return [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]


def split_test_files(test_patch: str) -> tuple[list[str], list[str]]:
    modified = []
    new = []
    for line in test_patch.strip().split("\n"):
        m = re.match(r"^diff --git a/(.+) b/(.+)$", line)
        if m:
            src, dst = m.group(1), m.group(2)
            if src == "/dev/null":
                new.append(dst)
            else:
                modified.append(src)
    return modified, new


def transform_test_directive(directive: str, repo: str) -> str:
    if repo == "django/django":
        d = directive[: -len(".py")] if directive.endswith(".py") else directive
        d = d[len("tests/") :] if d.startswith("tests/") else d
        return d.replace("/", ".")
    return directive


def get_test_cmd_template(repo: str, version: str) -> str:
    """Look up the canonical test invocation for a (repo, version) pair.

    Source: vendored MAP_REPO_VERSION_TO_SPECS_PY from the swebench package,
    which is the same table the official harness uses to build /eval.sh.
    Raises KeyError for unknown (repo, version) so missing coverage is loud.
    """
    return MAP_REPO_VERSION_TO_SPECS_PY[repo][version]["test_cmd"]


def build_test_command(repo: str, version: str, directives: list[str]) -> str:
    test_cmd = get_test_cmd_template(repo, version)
    args = [transform_test_directive(d, repo) for d in directives]
    return f"{CONDA_ACTIVATE} && cd /testbed && {test_cmd} {' '.join(args)}"


def update_results_file(results_path: Path, instance_id: str, data: dict):
    with _OUTPUT_FILE_LOCK:
        results = {}
        if results_path.exists():
            results = json.loads(results_path.read_text())
        results[instance_id] = data
        results_path.write_text(json.dumps(results, indent=2))


def process_instance_test_only(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    test_output_path = instance_dir / "test_output.txt"
    report_path = instance_dir / "report.json"

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Starting env")

    exit_status = "error"
    extra_info = {}

    try:
        progress_manager.update_instance_status(instance_id, "Pulling/starting container")
        env = get_sb_environment(config, instance)
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        test_patch = instance.get("test_patch", "")

        progress_manager.update_instance_status(instance_id, "Applying model patch")
        r = env.execute({"command": "cd /testbed && git apply -v patch.diff"})
        if r["returncode"] != 0:
            raise RuntimeError(f"patch apply failed: {r['output']}")

        directives = get_test_directives(test_patch)
        if not directives:
            logger.warning(f"{instance_id}: no test files found in test_patch")
            test_output = env.execute({"command": "echo NO_TEST_DIRECTIVES"})
            test_output_path.write_text(test_output.get("output", ""))
            exit_status = "no_tests"
        else:
            modified_files, new_files = split_test_files(test_patch)

            progress_manager.update_instance_status(instance_id, "Resetting test files")
            for f in modified_files:
                env.execute({"command": f"cd /testbed && git checkout {base_commit} -- {f}"})
            for f in new_files:
                env.execute({"command": f"cd /testbed && rm -f {f}"})

            progress_manager.update_instance_status(instance_id, "Applying test patch")
            b64 = base64.b64encode(test_patch.encode()).decode()
            env.execute(
                {
                    "command": f'python3 -c \'import base64; print(base64.b64decode("{b64}").decode(), end="")\' > /tmp/test.patch'
                }
            )
            r = env.execute({"command": "cd /testbed && git apply -v /tmp/test.patch"})
            if r["returncode"] != 0:
                raise RuntimeError(f"test patch apply failed: {r['output']}")

            progress_manager.update_instance_status(instance_id, "Running tests")
            test_cmd = build_test_command(repo, instance["version"], directives)
            t_start = time.monotonic()
            test_output = env.execute({"command": test_cmd, "timeout": 1800})
            test_duration = time.monotonic() - t_start
            extra_info["test_duration_seconds"] = round(test_duration, 2)
            test_output_path.write_text(test_output.get("output", ""))

            # Revert test files
            for f in modified_files:
                env.execute({"command": f"cd /testbed && git checkout {base_commit} -- {f}"})

            rc = test_output.get("returncode", -1)
            # rc 127 = command not found (test runner missing from PATH).
            # Sub-second runs are infrastructure failures, not real test results.
            if rc == 127 or (test_duration < 2.0 and rc != 0):
                exit_status = "env_error"
            elif rc == 0:
                exit_status = "resolved"
            else:
                exit_status = "unresolved"

        progress_manager.update_instance_status(instance_id, "Saving report")
        report = {
            instance_id: {
                "resolved": exit_status == "resolved",
                "exit_status": exit_status,
                "returncode": test_output.get("returncode", -1) if directives else -1,
                "test_directives": directives,
                "test_duration_seconds": round(test_duration, 2) if directives else None,
            }
        }
        report_path.write_text(json.dumps(report, indent=2))

    except Exception as e:
        logger.error(f"Error processing {instance_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        update_results_file(
            output_dir / "results.json",
            instance_id,
            {"exit_status": exit_status, "resolved": exit_status == "resolved", **extra_info},
        )
        progress_manager.on_instance_end(instance_id, exit_status)


# fmt: off
@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="Path to local SWE-bench dataset directory", rich_help_panel="Basic"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g. '0:5' for first 5)", rich_help_panel="Data selection"),
    output: str = typer.Option(..., "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Re-run instances already in results.json", rich_help_panel="Data selection"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (docker, singularity, etc.)", rich_help_panel="Advanced"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config files or key=value overrides (default: swebench.yaml)", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    """Apply patches and run test suites for SWE-bench instances (no agent)."""

    if not dataset:
        logger.error("--dataset is required")
        raise typer.Exit(1)

    dataset_dir = Path(dataset)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory '{dataset_dir}' does not exist")
        raise typer.Exit(1)

    dataset_metadata_path = dataset_dir / "logs" / "metadata.json"
    if not dataset_metadata_path.exists():
        logger.error("Invalid dataset directory: missing logs/metadata.json")
        raise typer.Exit(1)

    with open(dataset_metadata_path) as f:
        dataset_metadata = json.load(f)
        subset = dataset_metadata.get("subset", "").replace("swe-bench_", "")
        split = dataset_metadata.get("split", "")

    if "" in [subset, split]:
        logger.error(f"Dataset metadata incomplete (subset={subset!r}, split={split!r})")
        raise typer.Exit(1)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec="", slice_spec=slice_spec, shuffle=False)
    if not redo_existing and (output_path / "results.json").exists():
        existing = list(json.loads((output_path / "results.json").read_text()).keys())
        logger.info(f"Skipping {len(existing)} existing instances")
        instances = [i for i in instances if i["instance_id"] not in existing]
    logger.info(f"Running on {len(instances)} instances...")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "dataset": dataset_dir.absolute().as_posix(),
    })
    config = recursive_merge(*configs)

    with open(output_path / "run_config.json", "w") as f:
        json.dump({"dataset": dataset}, f)

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance_test_only, i, output_path, config, progress_manager): i[
                    "instance_id"
                ]
                for i in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
