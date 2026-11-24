"""Create a stopped production-quality clone without persisting private configuration."""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import socket
import subprocess
import urllib.parse
from typing import Any


def quality_command(command: list[str]) -> list[str]:
    """Change only speculative decoding and private API/collective ports."""
    result = list(command)
    index = result.index("--speculative-config")
    del result[index:index + 2]
    for flag, value in (("--port", "8047"), ("--master-port", "29729")):
        result[result.index(flag) + 1] = value
    return result


class DockerConnection(http.client.HTTPConnection):
    """Use the local Docker socket without writing environment secrets to disk."""

    def connect(self) -> None:
        """Connect through the local Unix socket."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/docker.sock")


def clone(original: str, name: str) -> dict[str, Any]:
    """Create an unstarted clone with the original image, mounts and all overlays."""
    value = json.loads(subprocess.check_output(["docker", "inspect", original]))[0]
    config = copy.deepcopy(value["Config"])
    config["Cmd"] = quality_command(config["Cmd"])
    config["Image"] = value["Image"]
    # Docker generates the new container hostname; host networking makes it immaterial.
    config.pop("Hostname", None)
    config["HostConfig"] = value["HostConfig"]
    existing = subprocess.run(["docker", "inspect", name], capture_output=True, check=False)
    if existing.returncode == 0:
        created = json.loads(existing.stdout)[0]
    else:
        connection = DockerConnection("localhost", timeout=60)
        connection.request("POST", "/containers/create?name=" + urllib.parse.quote(name),
                           body=json.dumps(config).encode(),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        if response.status != 201:
            raise RuntimeError(f"Quality clone creation failed with HTTP {response.status}")
        created = json.loads(subprocess.check_output(["docker", "inspect", payload["Id"]]))[0]
    if created["State"]["Running"] or created["Image"] != value["Image"]:
        raise RuntimeError("Quality preparation must not start or change the image")
    unchanged = set(value["Config"]) - {"Cmd", "Image", "Hostname"}
    if any(created["Config"].get(key) != value["Config"].get(key) for key in unchanged):
        raise RuntimeError("Unexpected production configuration change in quality clone")
    if created["Config"]["Cmd"] != quality_command(value["Config"]["Cmd"]):
        raise RuntimeError("Quality clone launch does not match the frozen command")
    old_host, new_host = copy.deepcopy(value["HostConfig"]), copy.deepcopy(created["HostConfig"])
    # Docker materializes its default null OOM flag as false on a newly created container.
    for host in (old_host, new_host):
        host["OomKillDisable"] = bool(host.get("OomKillDisable"))
    if new_host != old_host:
        raise RuntimeError("Unexpected production host setting change in quality clone")
    old_mounts = sorted(value["Mounts"], key=lambda m: m["Destination"])
    new_mounts = sorted(created["Mounts"], key=lambda m: m["Destination"])
    if old_mounts != new_mounts:
        raise RuntimeError("Production mounts were not preserved in quality clone")
    return {"id": created["Id"], "name": name, "image": created["Image"],
            "command": created["Config"]["Cmd"], "running": False,
            "configuration_changes": ["remove speculative config", "API port 8047",
                                      "collective port 29729", "new container hostname",
                                      "Docker normalizes default OomKillDisable null to false"],
            "unchanged_private_config_verified": True, "mounts_and_host_config_verified": True}


def main() -> None:
    """Prepare the user-authorized quality arm without starting any GPU process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(json.dumps(clone(args.original, args.name)))


if __name__ == "__main__":
    main()
