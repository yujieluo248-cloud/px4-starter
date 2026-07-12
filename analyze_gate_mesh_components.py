#!/usr/bin/env python3
"""
Read-only exact mesh-component inspector for PX4 Gazebo test_world.

It does NOT publish ROS 2 messages or send PX4 commands.

It:
  1) locates test_world -> test_world/model.sdf -> test_terrain.dae
  2) parses DAE triangle connectivity into independent mesh components
  3) writes a bounding-box row for each physical component
  4) saves one live Gazebo pose message from /world/test_world/pose/info

Use this instead of /world/.../scene/info, which may block in this Gazebo build.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


def local_name(elem):
    return elem.tag.rsplit("}", 1)[-1]


def nums(text):
    return [float(x) for x in (text or "").replace("\n", " ").split()]


def ints(text):
    return [int(x) for x in (text or "").replace("\n", " ").split()]


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def find_direct(parent, name):
    for c in list(parent):
        if local_name(c) == name:
            return c
    return None


def sources_for_mesh(mesh):
    result = {}
    for child in list(mesh):
        if local_name(child) != "source":
            continue
        sid = child.attrib.get("id", "")
        fa = find_direct(child, "float_array")
        tech = find_direct(child, "technique_common")
        acc = find_direct(tech, "accessor") if tech is not None else None
        if fa is None:
            continue
        stride = int(acc.attrib.get("stride", "3")) if acc is not None else 3
        arr = np.asarray(nums(fa.text), dtype=float)
        if stride < 3 or arr.size < stride:
            continue
        arr = arr[:(arr.size // stride) * stride].reshape((-1, stride))[:, :3]
        result[sid] = arr
    return result


def position_source_id(mesh):
    vertices = None
    for child in list(mesh):
        if local_name(child) == "vertices":
            vertices = child
            break
    if vertices is None:
        return None
    for inp in list(vertices):
        if local_name(inp) == "input" and inp.attrib.get("semantic") == "POSITION":
            return inp.attrib.get("source", "").lstrip("#")
    return None


def primitive_triangles(prim):
    """Return POSITION indices for triangle-like primitive, handling offsets."""
    inputs = [c for c in list(prim) if local_name(c) == "input"]
    if not inputs:
        return []

    stride = max(int(i.attrib.get("offset", "0")) for i in inputs) + 1
    vertex_offset = None
    for inp in inputs:
        semantic = inp.attrib.get("semantic")
        if semantic == "VERTEX":
            vertex_offset = int(inp.attrib.get("offset", "0"))
            break
    if vertex_offset is None:
        return []

    tag = local_name(prim)
    raw_p = []
    for c in list(prim):
        if local_name(c) == "p":
            raw_p.extend(ints(c.text))

    if tag == "triangles":
        tri_count = int(prim.attrib.get("count", "0"))
        needed = tri_count * 3 * stride
        raw_p = raw_p[:needed]
        out = []
        for i in range(0, len(raw_p), 3 * stride):
            if i + 3 * stride <= len(raw_p):
                out.append((
                    raw_p[i + vertex_offset],
                    raw_p[i + stride + vertex_offset],
                    raw_p[i + 2 * stride + vertex_offset],
                ))
        return out

    # polylist / polygons: triangulate simple polygons with fan method
    if tag == "polylist":
        vcount_elem = find_direct(prim, "vcount")
        if vcount_elem is None:
            return []
        vc = ints(vcount_elem.text)
        cursor = 0
        out = []
        for count in vc:
            polygon = []
            for j in range(count):
                idx = cursor + j * stride + vertex_offset
                if idx < len(raw_p):
                    polygon.append(raw_p[idx])
            cursor += count * stride
            for j in range(1, len(polygon) - 1):
                out.append((polygon[0], polygon[j], polygon[j + 1]))
        return out

    return []


def parse_dae_components(dae_path):
    root = ET.parse(dae_path).getroot()
    rows = []

    for geom in root.iter():
        if local_name(geom) != "geometry":
            continue
        mesh = find_direct(geom, "mesh")
        if mesh is None:
            continue

        pos_id = position_source_id(mesh)
        srcs = sources_for_mesh(mesh)
        if not pos_id or pos_id not in srcs:
            continue
        positions = srcs[pos_id]

        triangles = []
        for prim in list(mesh):
            if local_name(prim) in ("triangles", "polylist"):
                triangles.extend(primitive_triangles(prim))

        if not triangles:
            continue

        # Build connected components among vertices that share a triangle.
        used = sorted({i for tri in triangles for i in tri if 0 <= i < len(positions)})
        if not used:
            continue
        remap = {old: new for new, old in enumerate(used)}
        dsu = DSU(len(used))
        valid_tris = []
        for a, b, c in triangles:
            if a in remap and b in remap and c in remap:
                aa, bb, cc = remap[a], remap[b], remap[c]
                dsu.union(aa, bb)
                dsu.union(bb, cc)
                valid_tris.append((aa, bb, cc))

        groups = defaultdict(list)
        for old, new in remap.items():
            groups[dsu.find(new)].append(old)

        for component_index, old_indices in enumerate(groups.values(), start=1):
            pts = positions[np.asarray(old_indices, dtype=int)]
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            size = mx - mn
            rows.append({
                "geometry_id": geom.attrib.get("id", ""),
                "geometry_name": geom.attrib.get("name", geom.attrib.get("id", "")),
                "component": component_index,
                "vertex_count": len(old_indices),
                "min_x": float(mn[0]), "max_x": float(mx[0]),
                "min_y": float(mn[1]), "max_y": float(mx[1]),
                "min_z": float(mn[2]), "max_z": float(mx[2]),
                "size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2]),
            })

    return rows


def run(command, timeout=8):
    try:
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px4-root", default=str(Path.home() / "PX4-Autopilot"))
    ap.add_argument("--out", default="scene_inspection_exact")
    args = ap.parse_args()

    px4 = Path(args.px4_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    dae = px4 / "Tools/simulation/gz/models/test_world/meshes/test_terrain.dae"
    if not dae.exists():
        print(f"ERROR: missing {dae}", file=sys.stderr)
        return 2

    rows = parse_dae_components(dae)
    rows.sort(key=lambda r: (-r["size_z"], -r["vertex_count"]))

    with (out / "mesh_components.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    (out / "mesh_components.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Candidate physical obstacles: vertical components with nontrivial height.
    candidates = [
        r for r in rows
        if r["size_z"] >= 0.15 and r["vertex_count"] >= 12
    ]
    lines = [
        "# Exact DAE Connected Components",
        "",
        "Each row below is a connected triangle component from test_terrain.dae.",
        "These are local model coordinates. The test_world model pose is zero in the supplied world.",
        "",
        "## Vertical obstacle candidates",
        "",
        "| # | geometry | component | vertices | min_z | max_z | height | min_x | max_x | min_y | max_y |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, r in enumerate(candidates, start=1):
        lines.append(
            f"| {idx} | {r['geometry_name']} | {r['component']} | {r['vertex_count']} | "
            f"{r['min_z']:.3f} | {r['max_z']:.3f} | {r['size_z']:.3f} | "
            f"{r['min_x']:.3f} | {r['max_x']:.3f} | {r['min_y']:.3f} | {r['max_y']:.3f} |"
        )
    (out / "vertical_obstacle_candidates.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Read one live world pose message, no control commands.
    code, pose = run([
        "timeout", "5", "gz", "topic", "-e",
        "-t", "/world/test_world/pose/info", "-n", "1"
    ])
    (out / "live_pose_info.txt").write_text(pose, encoding="utf-8")

    summary = (
        f"Done.\n"
        f"DAE: {dae}\n"
        f"Connected components: {len(rows)}\n"
        f"Vertical candidates: {len(candidates)}\n"
        f"Output: {out}\n"
        f"Read-only live pose capture exit code: {code}\n"
    )
    (out / "README.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
