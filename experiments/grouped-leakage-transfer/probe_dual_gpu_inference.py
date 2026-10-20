"""Bounded inference-only preflight of two official local model workers.

No generated action is executed. No benchmark repair or model training runs.
The original single-GPU lock and results are preserved.
"""
from __future__ import annotations

import concurrent.futures
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import coding_agent as controller
from probe_local_inference import digest, extract, fetch, request_json


ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "local"
LOCK = ROOT / "inference-dual-gpu.lock.json"
REPORT = ROOT / "dual-gpu-preflight-result-v2.json"


def host_memory() -> dict:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot read host memory status")
    return {"total_physical_bytes": status.total_phys,
            "available_physical_bytes": status.avail_phys, "memory_load_percent": status.load}


def gpu_snapshot() -> list[str]:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version,compute_cap",
         "--format=csv,noheader,nounits"], text=True, timeout=15).strip().splitlines()


def probe_worker(worker: dict, settings: dict, deadline: float, result: dict) -> dict:
    result["measurements"] = []
    endpoint = f"http://{worker['host']}:{worker['port']}"
    client = controller.LocalModel(endpoint, {"max_http_response_bytes": 1048576}, deadline)
    for index in range(3):
        notes = ""
        if index == 2:
            notes = "\n".join(
                f"# note {line}: examples with the same group_id are related; inspect the implementation and preserve the evaluation checks."
                for line in range(384)) + "\n"
        messages = [
            {"role": "system", "content": "Return exactly one JSON read_file action, with scope repository and path src/split.py. Do not run any tools."},
            {"role": "user", "content": notes + "Inspect src/split.py."},
        ]
        started = time.monotonic()
        prompt, count = client.prepare(messages)
        response = client.complete(prompt, settings, 128)
        action = controller.parse_action(response["content"])
        expected = {"action": "read_file", "scope": "repository", "path": "src/split.py"}
        if action != expected or response.get("tokens_evaluated") != count:
            raise RuntimeError(f"{worker['id']}: unexpected action or token accounting")
        if response.get("truncated") or response.get("tokens_predicted", 129) > 128:
            raise RuntimeError(f"{worker['id']}: unexpected truncation or output allowance")
        measurement = {
            "repeat": index + 1, "long_prompt": index == 2,
            "wall_seconds": time.monotonic() - started, "prompt_tokens": count,
            "completion_tokens": response.get("tokens_predicted"),
            "timings": response.get("timings"), "action": action,
            "generated_action_executed": False,
        }
        result["measurements"].append(measurement)
        print(json.dumps({"worker": worker["id"], **measurement}), flush=True)
    result["status"] = "passed_inference_only"
    return result


def check_device_log(worker: dict, path: Path) -> dict:
    log = path.read_text(encoding="utf-8", errors="replace")
    if worker["device_name"].lower() not in log.lower():
        raise RuntimeError(f"{worker['id']}: expected device name missing from runtime log")
    offload = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", log, re.IGNORECASE)
    if not offload or int(offload[1]) != int(offload[2]) or int(offload[1]) <= 0:
        raise RuntimeError(f"{worker['id']}: complete GPU layer offload not verified")
    evidence = [line for line in log.splitlines()
                if worker["device_name"] in line or "offloaded" in line
                or ("CUDA" in line and "buffer" in line and ("KV" in line or "kv" in line))]
    if not any("buffer" in line and ("KV" in line or "kv" in line) for line in evidence):
        raise RuntimeError(f"{worker['id']}: CUDA KV allocation missing from runtime log")
    return {"expected_device_verified": True, "all_layers_offloaded": int(offload[1]),
            "cuda_kv_allocation_verified": True, "log_sha256_at_load": digest(path), "evidence": evidence}


def observe_safely(function) -> dict | list:
    try:
        return function()
    except Exception as error:
        return {"observation_error": f"{type(error).__name__}: {error}"}


def main() -> None:
    if REPORT.exists():
        raise RuntimeError(f"Refusing to overwrite existing evidence: {REPORT}")
    config = json.loads(LOCK.read_text(encoding="utf-8"))
    artifacts = [config["model"], *config["runtime"]["assets"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(fetch, artifacts))
    runtime = LOCAL / ("runtime-" + config["runtime"]["commit"][:12] + "-cuda-13.4")
    for archive in downloaded[1:]:
        extract(archive, runtime)
    servers = list(runtime.rglob("llama-server.exe"))
    if len(servers) != 1:
        raise RuntimeError("Expected exactly one official llama-server.exe")
    server = servers[0]
    for worker in config["workers"]:
        with socket.socket() as reservation:
            reservation.bind((worker["host"], worker["port"]))
    started = time.monotonic()
    deadline = started + 300
    report = {
        "schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting", "scope": "Native Windows dual-worker inference only; no generated code execution, repair success, or Linux Harness validation.",
        "inference_lock_file": LOCK.name, "inference_lock_sha256": digest(LOCK),
        "probe_script_sha256": digest(Path(__file__)),
        "effective_probe_limits": {"global_seconds_after_download": 300, "startup_seconds_per_worker": 120,
                                   "requests": "Bounded by the remaining absolute global deadline; no retry."},
        "legacy_probe_config_note": "The inherited lock probe section belongs to the original single-GPU probe; this script's actual caps are recorded above.",
        "artifacts_verified_by_size_and_sha256": True,
        "server_executable_sha256": digest(server),
        "model": config["model"], "runtime": config["runtime"],
        "inference": config["inference"], "gpu_before": gpu_snapshot(),
        "host_memory_before": host_memory(), "workers": [],
        "limitations": ["Short action generations are a compatibility check, not full repair throughput.",
                        "GPU-wide memory includes other processes; host availability is a snapshot.",
                        "No target ML pipeline or experience-building cycle has run."],
    }
    processes, logs = [], []
    try:
        settings = config["inference"]
        for worker in config["workers"]:
            if host_memory()["available_physical_bytes"] < 3 * 1024**3:
                raise RuntimeError("Less than 3 GiB host memory available before another model startup")
            environment = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
            environment["CUDA_VISIBLE_DEVICES"] = worker["gpu_uuid"]
            environment["PATH"] = os.pathsep.join([str(server.parent), str(runtime), environment.get("PATH", "")])
            command = [str(server), "--model", str(downloaded[0]), "--host", worker["host"],
                       "--port", str(worker["port"]), "--ctx-size", str(settings["context_tokens"]),
                       "--parallel", "1", "--n-gpu-layers", str(settings["gpu_layers"]),
                       "--split-mode", settings["split_mode"], "--threads", str(settings["threads"]),
                       "--threads-batch", str(settings["threads"]), "--batch-size", str(settings["batch_tokens"]),
                       "--ubatch-size", str(settings["microbatch_tokens"]),
                       "--flash-attn", settings["flash_attention"], "--cache-type-k", settings["key_cache_type"],
                       "--cache-type-v", settings["value_cache_type"], "--fit", "off", "--no-webui", "--no-context-shift", "--verbose"]
            log_path = LOCAL / f"dual-gpu-v2-{worker['id']}-server.log"
            log = log_path.open("wb")
            logs.append(log)
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            processes.append(process)
            worker_start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{worker['id']} exited {process.returncode}; inspect {log_path}")
                if time.monotonic() >= min(deadline, worker_start + 120):
                    raise TimeoutError(f"{worker['id']} startup exceeded the preflight limit")
                try:
                    health = request_json(f"http://{worker['host']}:{worker['port']}/health", timeout=1)
                    if health.get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.25)
            report["workers"].append({"worker": worker, "startup_seconds": time.monotonic() - worker_start,
                                      "device_verification": check_device_log(worker, log_path),
                                      "properties": request_json(f"http://{worker['host']}:{worker['port']}/props", timeout=5)})
            print(json.dumps({"worker": worker["id"], "status": "loaded", "host_memory": host_memory()}), flush=True)
        report["gpu_both_loaded"] = gpu_snapshot()
        report["host_memory_both_loaded"] = host_memory()
        if report["host_memory_both_loaded"]["available_physical_bytes"] < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB host memory headroom with both workers loaded")
        concurrent_start = time.monotonic()
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(probe_worker, entry["worker"], settings, deadline, entry): entry
                       for entry in report["workers"]}
            for future in concurrent.futures.as_completed(futures):
                entry = futures[future]
                try:
                    future.result()
                except Exception as error:
                    entry["status"] = "failed_inference_probe"
                    entry["error"] = f"{type(error).__name__}: {error}"
                    failures.append(entry["worker"]["id"])
        report["concurrent_probe_wall_seconds"] = time.monotonic() - concurrent_start
        if failures:
            raise RuntimeError(f"Worker inference probes failed: {failures}; partial measurements retained")
        report["gpu_after_requests"] = gpu_snapshot()
        report["host_memory_after_requests"] = host_memory()
        report["status"] = "passed_dual_gpu_native_inference_only"
    except Exception as error:
        report["status"], report["error"] = "failed_dual_gpu_preflight", f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        for process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            except Exception as error:
                cleanup_errors.append(str(error))
        for log in logs:
            log.close()
        report["cleanup_errors"] = cleanup_errors
        report["total_preflight_seconds_excluding_download"] = time.monotonic() - started
        report["host_memory_after_cleanup"] = observe_safely(host_memory)
        report["gpu_after_cleanup"] = observe_safely(gpu_snapshot)
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Evidence: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
