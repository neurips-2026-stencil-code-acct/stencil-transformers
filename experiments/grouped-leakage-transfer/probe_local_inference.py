"""Verify pinned public artifacts and run an inference-only native preflight.

This script never executes generated code or starts a benchmark/training task.
Downloaded official inference binaries are the only executables it launches.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "inference.lock.json"
LOCAL = ROOT / "local"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(artifact: dict) -> Path:
    target = LOCAL / "downloads" / artifact["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == artifact["size_bytes"] and digest(target) == artifact["sha256"]:
            print(f"Verified existing {target.name}", flush=True)
            return target
        raise RuntimeError(f"Existing artifact failed verification: {target}")
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(artifact["url"], headers={"User-Agent": "OpenRSI-local-model-preflight/1"})
    value = hashlib.sha256()
    count = 0
    started = last_report = time.monotonic()
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        while block := response.read(4 * 1024 * 1024):
            count += len(block)
            value.update(block)
            output.write(block)
            now = time.monotonic()
            if now - last_report >= 20:
                print(f"{target.name}: {count / 1e9:.2f}/{artifact['size_bytes'] / 1e9:.2f} GB, {count / max(now - started, 1) / 1e6:.1f} MB/s", flush=True)
                last_report = now
    if count != artifact["size_bytes"] or value.hexdigest() != artifact["sha256"]:
        raise RuntimeError(f"Downloaded artifact failed size/SHA256 verification: {temporary}")
    temporary.replace(target)
    print(f"Verified {target.name} ({count:,} bytes)", flush=True)
    return target


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    boundary = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            resolved = (destination / entry.filename).resolve()
            if not resolved.is_relative_to(boundary):
                raise RuntimeError(f"Archive member escapes runtime directory: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"Archive contains a symlink: {entry.filename}")
        bundle.extractall(destination)


def request_json(url: str, payload: dict | None = None, timeout: float = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(LOCK.read_text(encoding="utf-8"))
    artifacts = [config["model"], *config["runtime"]["assets"]]
    needed = sum(item["size_bytes"] for item in artifacts)
    if shutil.disk_usage(ROOT).free < needed * 2:
        raise RuntimeError("Insufficient disk headroom for the verified downloads and extraction")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, artifacts))
    if args.download_only:
        return
    runtime = LOCAL / ("runtime-" + config["runtime"]["commit"][:12])
    for archive in files[1:]:
        extract(archive, runtime)
    servers = list(runtime.rglob("llama-server.exe"))
    if len(servers) != 1:
        raise RuntimeError(f"Expected exactly one llama-server.exe; found {len(servers)}")
    server = servers[0]
    model = files[0]
    settings = config["inference"]
    probe = config["probe"]
    device_info = subprocess.check_output(
        ["nvidia-smi", "-i", settings["cuda_visible_devices"],
         "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"], text=True).strip()
    if not device_info.startswith(settings["required_device_name"] + ","):
        raise RuntimeError(f"Unexpected selected device: {device_info}")
    with socket.socket() as available_port:
        available_port.bind((probe["host"], probe["port"]))
    environment = {name: value for name, value in os.environ.items() if not name.startswith("LLAMA_ARG_")}
    environment["CUDA_VISIBLE_DEVICES"] = settings["cuda_visible_devices"]
    environment["PATH"] = os.pathsep.join([str(server.parent), str(runtime), environment.get("PATH", "")])
    command = [str(server), "--model", str(model), "--host", probe["host"], "--port", str(probe["port"]),
               "--ctx-size", str(settings["context_tokens"]), "--parallel", str(settings["parallel_slots"]),
               "--n-gpu-layers", str(settings["gpu_layers"]), "--split-mode", settings["split_mode"],
               "--threads", str(settings["threads"]), "--batch-size", str(settings["batch_tokens"]),
               "--ubatch-size", str(settings["microbatch_tokens"]), "--flash-attn", settings["flash_attention"],
               "--cache-type-k", settings["key_cache_type"], "--cache-type-v", settings["value_cache_type"],
               "--fit", "off", "--no-webui", "--no-context-shift"]
    report = {"status": "starting", "lock_sha256": digest(LOCK), "command": command, "measurements": [],
              "device_before_inference": device_info, "cuda_visible_devices": settings["cuda_visible_devices"],
              "scope": "Inference-only preflight; no generated code executed; no benchmark repair-success measurement."}
    LOCAL.mkdir(parents=True, exist_ok=True)
    report_path = LOCAL / "inference-probe.json"
    log_path = LOCAL / "llama-server.log"
    started = time.monotonic()
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            endpoint = f"http://{probe['host']}:{probe['port']}"
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with {process.returncode}; inspect {log_path}")
                try:
                    if request_json(endpoint + "/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() - started > probe["startup_timeout_seconds"]:
                    raise TimeoutError("Server startup exceeded the probe limit")
                time.sleep(0.5)
            report["startup_seconds"] = time.monotonic() - started
            report["effective_properties"] = request_json(endpoint + "/props")
            report["device_after_context_allocation"] = subprocess.check_output(
                ["nvidia-smi", "-i", settings["cuda_visible_devices"],
                 "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
            schema = {"type": "object", "properties": {"action": {"type": "string", "enum": ["read_file"]},
                      "path": {"type": "string"}}, "required": ["action", "path"], "additionalProperties": False}
            for index in range(3):
                notes = ""
                if index == 2:
                    notes = "\n".join(f"# note {line}: examples with the same group_id are related; inspect the implementation and preserve the evaluation checks."
                                      for line in range(probe["long_prompt_note_lines"])) + "\n"
                payload = {"model": model.stem,
                           "messages": [{"role": "system", "content": "You are a coding agent. Return exactly one JSON file-reading action. Do not execute anything."},
                                        {"role": "user", "content": notes + "A Python ML repository has data splitting logic in src/split.py. Select a file-reading action to inspect it."}],
                           "temperature": settings["temperature"], "seed": settings["seed"],
                           "top_p": settings["top_p"], "top_k": settings["top_k"], "min_p": settings["min_p"],
                           "repeat_penalty": settings["repeat_penalty"], "cache_prompt": settings["cache_prompt"],
                           "max_tokens": probe["maximum_new_tokens"],
                           "response_format": {"type": "json_schema", "json_schema": {"name": "read_action", "strict": True, "schema": schema}}}
                request_started = time.monotonic()
                result = request_json(endpoint + "/v1/chat/completions", payload, probe["request_timeout_seconds"])
                elapsed = time.monotonic() - request_started
                content = result["choices"][0]["message"]["content"]
                action = json.loads(content)
                if action != {"action": "read_file", "path": "src/split.py"}:
                    raise RuntimeError(f"Unexpected constrained response: {action}")
                measurement = {"repeat": index + 1, "long_prompt_probe": index == 2,
                               "wall_seconds": elapsed, "usage": result.get("usage"),
                               "timings": result.get("timings"), "response": action, "finish_reason": result["choices"][0].get("finish_reason")}
                report["measurements"].append(measurement)
                print(json.dumps(measurement), flush=True)
            report["status"] = "passed_inference_probe"
            controller_path = ROOT / "coding_agent.py"
            if controller_path.is_file():
                spec = importlib.util.spec_from_file_location("coding_agent_probe", controller_path)
                controller = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(controller)
                client = controller.LocalModel(endpoint, {"max_http_response_bytes": 1048576},
                                               time.monotonic() + probe["request_timeout_seconds"])
                messages = [{"role": "system", "content": "Return one JSON read_file action, with scope repository and path src/split.py. Do not run any tools."},
                            {"role": "user", "content": "Inspect src/split.py."}]
                prompt, prompt_tokens = client.prepare(messages)
                reply = client.complete(prompt, settings, probe["maximum_new_tokens"])
                action = controller.parse_action(reply["content"])
                if action != {"action": "read_file", "scope": "repository", "path": "src/split.py"}:
                    raise RuntimeError(f"Unexpected controller adapter response: {action}")
                report["controller_adapter_probe"] = {"status": "passed", "controller_sha256": digest(controller_path),
                                                       "prompt_tokens": prompt_tokens, "evaluated_tokens": reply.get("tokens_evaluated"),
                                                       "timings": reply.get("timings"),
                                                       "response": action, "generated_action_executed": False}
    except Exception as error:
        report["status"] = "failed_inference_probe"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        report["total_probe_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Probe report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
