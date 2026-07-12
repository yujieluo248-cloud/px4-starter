#!/usr/bin/env python3
"""
Read-only Gazebo / PX4 scene inspector.

What it does:
1. Parses the active test_world SDF and its referenced model SDF files.
2. Parses COLLADA (.dae) meshes used by the scene.
3. Produces bounding boxes for every DAE scene geometry instance:
      min/max X, Y, Z in the DAE world frame.
4. Captures Gazebo's currently advertised topics / services.
5. Attempts to capture one live pose-info message from Gazebo.

What it does NOT do:
- Does not send ROS 2 messages.
- Does not arm, take off, move, land, or modify the Gazebo world.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def find_children(parent: ET.Element, name: str) -> List[ET.Element]:
    return [child for child in list(parent) if tag_name(child) == name]


def find_first(parent: ET.Element, name: str) -> Optional[ET.Element]:
    for child in parent.iter():
        if tag_name(child) == name:
            return child
    return None


def parse_numbers(text: Optional[str]) -> List[float]:
    if not text:
        return []
    return [float(x) for x in text.replace("\n", " ").split()]


def parse_pose(text: Optional[str]) -> Tuple[float, float, float, float, float, float]:
    values = parse_numbers(text)
    if len(values) < 6:
        values += [0.0] * (6 - len(values))
    return tuple(values[:6])  # x y z roll pitch yaw


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def sdf_pose_matrix(text: Optional[str]) -> np.ndarray:
    x, y, z, roll, pitch, yaw = parse_pose(text)
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m @ rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def translation_matrix(v: List[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[:3, 3] = v[:3]
    return m


def scale_matrix(v: List[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[0, 0], m[1, 1], m[2, 2] = v[:3]
    return m


def axis_angle_matrix(v: List[float]) -> np.ndarray:
    if len(v) < 4:
        return np.eye(4)
    x, y, z, deg = v[:4]
    axis = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(4)
    x, y, z = axis / norm
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    one_c = 1 - c
    r = np.array([
        [c + x*x*one_c, x*y*one_c - z*s, x*z*one_c + y*s],
        [y*x*one_c + z*s, c + y*y*one_c, y*z*one_c - x*s],
        [z*x*one_c - y*s, z*y*one_c + x*s, c + z*z*one_c],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    return m


def collada_node_matrix(node: ET.Element) -> np.ndarray:
    """Apply COLLADA transforms in document order."""
    m = np.eye(4)
    for child in list(node):
        name = tag_name(child)
        values = parse_numbers(child.text)
        if name == "matrix" and len(values) >= 16:
            # COLLADA matrices are column-major.
            local = np.array(values[:16], dtype=float).reshape((4, 4), order="F")
        elif name == "translate":
            local = translation_matrix(values)
        elif name == "scale":
            local = scale_matrix(values)
        elif name == "rotate":
            local = axis_angle_matrix(values)
        else:
            continue
        m = m @ local
    return m


@dataclass
class BoundRow:
    dae_file: str
    node_path: str
    node_id: str
    node_name: str
    geometry_id: str
    geometry_name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    size_x: float
    size_y: float
    size_z: float


def geometry_position_bounds(dae_root: ET.Element) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    """Return local POSITION bounds per geometry id."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}

    for geometry in dae_root.iter():
        if tag_name(geometry) != "geometry":
            continue

        geom_id = geometry.attrib.get("id", "")
        geom_name = geometry.attrib.get("name", geom_id)

        mesh = find_first(geometry, "mesh")
        if mesh is None:
            continue

        vertices = find_first(mesh, "vertices")
        if vertices is None:
            continue

        position_source_id = None
        for inp in list(vertices):
            if tag_name(inp) == "input" and inp.attrib.get("semantic") == "POSITION":
                position_source_id = inp.attrib.get("source", "").lstrip("#")
                break
        if not position_source_id:
            continue

        source = None
        for cand in list(mesh):
            if tag_name(cand) == "source" and cand.attrib.get("id") == position_source_id:
                source = cand
                break
        if source is None:
            continue

        float_array = find_first(source, "float_array")
        accessor = find_first(source, "accessor")
        if float_array is None:
            continue

        values = parse_numbers(float_array.text)
        stride = 3
        if accessor is not None:
            try:
                stride = int(accessor.attrib.get("stride", "3"))
            except ValueError:
                stride = 3
        if stride < 3 or len(values) < 3:
            continue

        arr = np.array(values, dtype=float)
        count = len(arr) // stride
        arr = arr[:count * stride].reshape((count, stride))[:, :3]
        if len(arr) == 0:
            continue

        out[geom_id] = (arr.min(axis=0), arr.max(axis=0), geom_name)

    return out


def transform_bbox(min_pt: np.ndarray, max_pt: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corners = np.array([
        [x, y, z, 1.0]
        for x in (min_pt[0], max_pt[0])
        for y in (min_pt[1], max_pt[1])
        for z in (min_pt[2], max_pt[2])
    ])
    transformed = (m @ corners.T).T[:, :3]
    return transformed.min(axis=0), transformed.max(axis=0)


def scene_geometry_instances(
    dae_root: ET.Element,
    dae_file: Path,
    parent_sdf_matrix: np.ndarray,
) -> List[BoundRow]:
    geom_bounds = geometry_position_bounds(dae_root)
    rows: List[BoundRow] = []

    visual_scene = None
    for elem in dae_root.iter():
        if tag_name(elem) == "visual_scene":
            visual_scene = elem
            break
    if visual_scene is None:
        return rows

    def walk(node: ET.Element, parent_matrix: np.ndarray, path: List[str]) -> None:
        node_id = node.attrib.get("id", "")
        node_name = node.attrib.get("name", node_id or "unnamed")
        current_path = path + [node_name]
        world_m = parent_matrix @ collada_node_matrix(node)

        for child in list(node):
            if tag_name(child) == "instance_geometry":
                geom_id = child.attrib.get("url", "").lstrip("#")
                if geom_id not in geom_bounds:
                    continue
                local_min, local_max, geom_name = geom_bounds[geom_id]
                world_min, world_max = transform_bbox(local_min, local_max, world_m)
                size = world_max - world_min
                rows.append(BoundRow(
                    dae_file=str(dae_file),
                    node_path="/".join(current_path),
                    node_id=node_id,
                    node_name=node_name,
                    geometry_id=geom_id,
                    geometry_name=geom_name,
                    min_x=float(world_min[0]),
                    max_x=float(world_max[0]),
                    min_y=float(world_min[1]),
                    max_y=float(world_max[1]),
                    min_z=float(world_min[2]),
                    max_z=float(world_max[2]),
                    size_x=float(size[0]),
                    size_y=float(size[1]),
                    size_z=float(size[2]),
                ))
            elif tag_name(child) == "node":
                walk(child, world_m, current_path)

    for child in list(visual_scene):
        if tag_name(child) == "node":
            walk(child, parent_sdf_matrix, [])

    return rows


def model_paths_from_world(world_sdf: Path, px4_root: Path) -> List[Tuple[str, Path, np.ndarray]]:
    tree = ET.parse(world_sdf)
    root = tree.getroot()
    found: List[Tuple[str, Path, np.ndarray]] = []

    for include in root.iter():
        if tag_name(include) != "include":
            continue

        uri_elem = next((c for c in list(include) if tag_name(c) == "uri"), None)
        name_elem = next((c for c in list(include) if tag_name(c) == "name"), None)
        pose_elem = next((c for c in list(include) if tag_name(c) == "pose"), None)

        if uri_elem is None or not uri_elem.text:
            continue

        uri = uri_elem.text.strip()
        if not uri.startswith("model://"):
            continue

        model_name = uri.removeprefix("model://")
        name = name_elem.text.strip() if name_elem is not None and name_elem.text else model_name
        model_sdf = px4_root / "Tools/simulation/gz/models" / model_name / "model.sdf"

        if model_sdf.exists():
            found.append((name, model_sdf, sdf_pose_matrix(pose_elem.text if pose_elem is not None else None)))

    return found


def parse_sdf_meshes(model_sdf: Path, model_world_pose: np.ndarray) -> List[Tuple[Path, np.ndarray]]:
    """Return mesh paths and their world transform from an SDF model."""
    tree = ET.parse(model_sdf)
    root = tree.getroot()
    result: List[Tuple[Path, np.ndarray]] = []

    # Model pose may exist inside model.sdf too.
    model_pose_elem = next((e for e in root.iter() if tag_name(e) == "model"), None)
    internal_model_pose = np.eye(4)
    if model_pose_elem is not None:
        pose = next((c for c in list(model_pose_elem) if tag_name(c) == "pose"), None)
        internal_model_pose = sdf_pose_matrix(pose.text if pose is not None else None)

    base = model_world_pose @ internal_model_pose

    for link in root.iter():
        if tag_name(link) != "link":
            continue

        link_pose_elem = next((c for c in list(link) if tag_name(c) == "pose"), None)
        link_pose = sdf_pose_matrix(link_pose_elem.text if link_pose_elem is not None else None)

        for visual_or_collision in list(link):
            if tag_name(visual_or_collision) not in ("visual", "collision"):
                continue

            vc_pose_elem = next((c for c in list(visual_or_collision) if tag_name(c) == "pose"), None)
            vc_pose = sdf_pose_matrix(vc_pose_elem.text if vc_pose_elem is not None else None)

            mesh_uri = None
            for elem in visual_or_collision.iter():
                if tag_name(elem) == "uri" and elem.text:
                    mesh_uri = elem.text.strip()
                    break

            if not mesh_uri or not mesh_uri.startswith("model://"):
                continue

            rest = mesh_uri.removeprefix("model://")
            parts = rest.split("/", 1)
            if len(parts) != 2:
                continue
            mesh_path = model_sdf.parent.parent / parts[0] / parts[1]
            if mesh_path.exists() and mesh_path.suffix.lower() == ".dae":
                result.append((mesh_path, base @ link_pose @ vc_pose))

    return result


def run_command(command: List[str], timeout: int = 8) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}\n"


def safe_write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Gazebo scene inspector")
    parser.add_argument(
        "--px4-root",
        default=str(Path.home() / "PX4-Autopilot"),
        help="PX4-Autopilot directory",
    )
    parser.add_argument(
        "--world",
        default="test_world",
        help="World SDF basename without .sdf",
    )
    parser.add_argument(
        "--out",
        default="scene_inspection",
        help="Output directory",
    )
    args = parser.parse_args()

    px4_root = Path(args.px4_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    world_sdf = px4_root / "Tools/simulation/gz/worlds" / f"{args.world}.sdf"
    if not world_sdf.exists():
        print(f"ERROR: world file not found: {world_sdf}", file=sys.stderr)
        return 2

    report: Dict[str, object] = {
        "px4_root": str(px4_root),
        "world_sdf": str(world_sdf),
        "world_name": args.world,
        "models": [],
        "geometry_bounds": [],
    }

    all_rows: List[BoundRow] = []
    models = model_paths_from_world(world_sdf, px4_root)

    for instance_name, model_sdf, world_pose in models:
        mesh_entries = parse_sdf_meshes(model_sdf, world_pose)
        model_record = {
            "instance_name": instance_name,
            "model_sdf": str(model_sdf),
            "mesh_count": len(mesh_entries),
            "meshes": [],
        }

        for mesh_path, mesh_matrix in mesh_entries:
            try:
                dae_root = ET.parse(mesh_path).getroot()
                rows = scene_geometry_instances(dae_root, mesh_path, mesh_matrix)
                all_rows.extend(rows)
                model_record["meshes"].append({
                    "mesh_path": str(mesh_path),
                    "geometry_instances": len(rows),
                })
            except Exception as exc:
                model_record["meshes"].append({
                    "mesh_path": str(mesh_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })

        report["models"].append(model_record)

    report["geometry_bounds"] = [asdict(r) for r in all_rows]
    safe_write(out_dir / "scene_geometry.json", json.dumps(report, indent=2, ensure_ascii=False))

    if all_rows:
        with (out_dir / "dae_geometry_bounds.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
            writer.writeheader()
            for row in all_rows:
                writer.writerow(asdict(row))

    # Human readable condensed table sorted by height then name.
    condensed = sorted(all_rows, key=lambda r: (-r.size_z, r.node_path))
    lines = [
        "# Static Geometry Bounds",
        "",
        "These values are read from the actual SDF / DAE scene files.",
        "They are in the DAE / model world coordinates after SDF poses are applied.",
        "",
        "| Node path | min_z | max_z | height | min_x | max_x | min_y | max_y |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in condensed:
        lines.append(
            f"| {r.node_path} | {r.min_z:.3f} | {r.max_z:.3f} | {r.size_z:.3f} | "
            f"{r.min_x:.3f} | {r.max_x:.3f} | {r.min_y:.3f} | {r.max_y:.3f} |"
        )
    safe_write(out_dir / "scene_geometry.md", "\n".join(lines) + "\n")

    # Live Gazebo information. This is read-only.
    live_lines = []
    gz = shutil.which("gz")
    if not gz:
        live_lines.append("gz command not found in PATH.\n")
    else:
        for title, cmd in [
            ("gz topic -l", [gz, "topic", "-l"]),
            ("gz service -l", [gz, "service", "-l"]),
        ]:
            code, output = run_command(cmd)
            live_lines.append(f"\n===== {title} (exit {code}) =====\n{output}")

        code, topics = run_command([gz, "topic", "-l"])
        pose_topics = []
        if code == 0:
            pose_topics = [
                line.strip()
                for line in topics.splitlines()
                if "/pose/info" in line.strip()
            ]

        live_lines.append(
            "\n===== detected pose topics =====\n"
            + ("\n".join(pose_topics) if pose_topics else "(none found)")
            + "\n"
        )

        for topic in pose_topics[:3]:
            # gz topic -e reads only. timeout prevents a blocked command.
            cmd = ["timeout", "6", gz, "topic", "-e", "-t", topic, "-n", "1"]
            code, output = run_command(cmd, timeout=8)
            live_lines.append(
                f"\n===== one live pose message from {topic} (exit {code}) =====\n{output}"
            )

    safe_write(out_dir / "live_gazebo_introspection.txt", "\n".join(live_lines))

    summary = textwrap.dedent(f"""\
        Inspection complete.

        Output folder: {out_dir}

        Main files:
          - scene_geometry.md        readable geometry table
          - dae_geometry_bounds.csv  full geometry bounds for Excel / filtering
          - scene_geometry.json      machine-readable complete result
          - live_gazebo_introspection.txt
                                    current Gazebo topics, services, and live pose message

        This script sent no vehicle-control commands.
    """)
    safe_write(out_dir / "README.txt", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
