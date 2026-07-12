#!/usr/bin/env python3
"""
Fast, read-only left-frame opening extractor for PX4 Gazebo test_world.

Why this version is fast
-------------------------
The earlier version performed a ray-vs-triangle intersection for every grid
cell. This version projects the already transformed left-frame triangles onto
the X-Z plane and rasterizes that silhouette with OpenCV in milliseconds.

Result
------
It detects enclosed free regions inside the left frame silhouette and reports:
  - opening min/max X and Z
  - opening width / height
  - opening center (X, Y, Z)
  - an image and text report for visual verification

Safety
------
Read-only:
- no ROS 2 publishers
- no PX4 commands
- no Gazebo services
- no model movement
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print(
        "ERROR: This script needs OpenCV. Install it once with:\n"
        "  sudo apt install -y python3-opencv\n",
        file=sys.stderr,
    )
    raise SystemExit(2)


# ------------------------------- XML helpers -------------------------------

def lname(e: ET.Element) -> str:
    return e.tag.rsplit("}", 1)[-1]


def direct(parent: ET.Element | None, name: str) -> ET.Element | None:
    if parent is None:
        return None
    for c in list(parent):
        if lname(c) == name:
            return c
    return None


def fnums(text: str | None) -> list[float]:
    return [float(x) for x in (text or "").replace("\n", " ").split()]


def inums(text: str | None) -> list[int]:
    return [int(x) for x in (text or "").replace("\n", " ").split()]


# ------------------------------ transforms ----------------------------------

def translation(v: list[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[:3, 3] = v[:3]
    return m


def scaling(v: list[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[0, 0], m[1, 1], m[2, 2] = v[:3]
    return m


def axis_angle(v: list[float]) -> np.ndarray:
    if len(v) < 4:
        return np.eye(4)
    axis = np.asarray(v[:3], dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(4)

    x, y, z = axis / n
    a = math.radians(v[3])
    c, s = math.cos(a), math.sin(a)
    q = 1.0 - c

    r = np.array([
        [c + x*x*q,   x*y*q - z*s, x*z*q + y*s],
        [y*x*q + z*s, c + y*y*q,   y*z*q - x*s],
        [z*x*q - y*s, z*y*q + x*s, c + z*z*q],
    ])

    m = np.eye(4)
    m[:3, :3] = r
    return m


def node_transform(node: ET.Element) -> np.ndarray:
    m = np.eye(4)

    for c in list(node):
        name = lname(c)
        vals = fnums(c.text)

        if name == "matrix" and len(vals) >= 16:
            local = np.asarray(vals[:16], dtype=float).reshape((4, 4), order="F")
        elif name == "translate":
            local = translation(vals)
        elif name == "scale":
            local = scaling(vals)
        elif name == "rotate":
            local = axis_angle(vals)
        else:
            continue

        m = m @ local

    return m


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    h = np.c_[points, np.ones(len(points))]
    return (m @ h.T).T[:, :3]


# ------------------------------- DAE parsing --------------------------------

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def read_sources(mesh: ET.Element) -> tuple[dict[str, np.ndarray], str | None]:
    sources: dict[str, np.ndarray] = {}

    for src in list(mesh):
        if lname(src) != "source":
            continue

        source_id = src.attrib.get("id", "")
        float_array = direct(src, "float_array")
        tech = direct(src, "technique_common")
        accessor = direct(tech, "accessor")

        if float_array is None:
            continue

        stride = int(accessor.attrib.get("stride", "3")) if accessor is not None else 3
        values = np.asarray(fnums(float_array.text), dtype=float)

        if stride < 3 or values.size < stride:
            continue

        values = values[:(values.size // stride) * stride]
        sources[source_id] = values.reshape((-1, stride))[:, :3]

    vertices = direct(mesh, "vertices")
    position_source = None

    if vertices is not None:
        for inp in list(vertices):
            if lname(inp) == "input" and inp.attrib.get("semantic") == "POSITION":
                position_source = inp.attrib.get("source", "").lstrip("#")
                break

    return sources, position_source


def triangles_from_primitive(prim: ET.Element) -> list[tuple[int, int, int]]:
    inputs = [c for c in list(prim) if lname(c) == "input"]
    if not inputs:
        return []

    stride = max(int(i.attrib.get("offset", "0")) for i in inputs) + 1
    vertex_offset = None

    for inp in inputs:
        if inp.attrib.get("semantic") == "VERTEX":
            vertex_offset = int(inp.attrib.get("offset", "0"))
            break

    if vertex_offset is None:
        return []

    raw: list[int] = []
    for c in list(prim):
        if lname(c) == "p":
            raw.extend(inums(c.text))

    kind = lname(prim)

    if kind == "triangles":
        count = int(prim.attrib.get("count", "0"))
        raw = raw[:count * 3 * stride]
        return [
            (raw[i + vertex_offset], raw[i + stride + vertex_offset], raw[i + 2 * stride + vertex_offset])
            for i in range(0, len(raw), 3 * stride)
            if i + 3 * stride <= len(raw)
        ]

    if kind == "polylist":
        vcount = direct(prim, "vcount")
        if vcount is None:
            return []

        out: list[tuple[int, int, int]] = []
        cursor = 0

        for n in inums(vcount.text):
            polygon = [
                raw[cursor + j * stride + vertex_offset]
                for j in range(n)
                if cursor + j * stride + vertex_offset < len(raw)
            ]
            cursor += n * stride

            for j in range(1, len(polygon) - 1):
                out.append((polygon[0], polygon[j], polygon[j + 1]))

        return out

    return []


def load_geometry(root: ET.Element) -> dict[str, dict]:
    result: dict[str, dict] = {}

    for geo in root.iter():
        if lname(geo) != "geometry":
            continue

        mesh = direct(geo, "mesh")
        if mesh is None:
            continue

        sources, pos_id = read_sources(mesh)
        if not pos_id or pos_id not in sources:
            continue

        tris: list[tuple[int, int, int]] = []
        for primitive in list(mesh):
            if lname(primitive) in ("triangles", "polylist"):
                tris.extend(triangles_from_primitive(primitive))

        if tris:
            result[geo.attrib.get("id", "")] = {
                "name": geo.attrib.get("name", geo.attrib.get("id", "")),
                "positions": sources[pos_id],
                "triangles": tris,
            }

    return result


def connected_groups(position_count: int, tris: list[tuple[int, int, int]]) -> list[set[int]]:
    valid = [t for t in tris if all(0 <= i < position_count for i in t)]
    used = sorted({i for tri in valid for i in tri})
    if not used:
        return []

    remap = {old: new for new, old in enumerate(used)}
    dsu = DSU(len(used))

    for a, b, c in valid:
        dsu.union(remap[a], remap[b])
        dsu.union(remap[b], remap[c])

    groups: dict[int, set[int]] = defaultdict(set)
    for old, new in remap.items():
        groups[dsu.find(new)].add(old)

    return list(groups.values())


def gather_instances(root: ET.Element, geometries: dict[str, dict]) -> list[dict]:
    scene = next((e for e in root.iter() if lname(e) == "visual_scene"), None)
    if scene is None:
        return []

    instances: list[dict] = []

    def walk(node: ET.Element, parent_m: np.ndarray, path: list[str]) -> None:
        node_name = node.attrib.get("name", node.attrib.get("id", "node"))
        current_m = parent_m @ node_transform(node)
        current_path = path + [node_name]

        for child in list(node):
            child_name = lname(child)

            if child_name == "instance_geometry":
                gid = child.attrib.get("url", "").lstrip("#")
                if gid not in geometries:
                    continue

                data = geometries[gid]
                positions = data["positions"]
                tris = data["triangles"]

                for component_id, group in enumerate(connected_groups(len(positions), tris), start=1):
                    ids = np.asarray(sorted(group), dtype=int)
                    world_points = transform_points(positions[ids], current_m)
                    mn, mx = world_points.min(axis=0), world_points.max(axis=0)
                    group_tris = [t for t in tris if t[0] in group and t[1] in group and t[2] in group]

                    instances.append({
                        "path": "/".join(current_path),
                        "geometry_name": data["name"],
                        "component_id": component_id,
                        "positions": positions,
                        "triangles": group_tris,
                        "matrix": current_m,
                        "bounds_min": mn,
                        "bounds_max": mx,
                    })

            elif child_name == "node":
                walk(child, current_m, current_path)

    for child in list(scene):
        if lname(child) == "node":
            walk(child, np.eye(4), [])

    return instances


# ------------------------- fast opening extraction --------------------------

def choose_left_frame(instances: list[dict]) -> dict:
    target = np.array([-0.060, 1.140, 3.078, 3.378, 0.000, 1.853])
    best, best_score = None, float("inf")

    for inst in instances:
        mn, mx = inst["bounds_min"], inst["bounds_max"]
        actual = np.array([mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]])
        score = float(np.linalg.norm(actual - target))

        if score < best_score:
            best, best_score = inst, score

    if best is None:
        raise RuntimeError("Unable to locate the left frame component.")

    return best


def component_world_triangles(inst: dict) -> np.ndarray:
    pts = inst["positions"]
    m = inst["matrix"]

    out = []
    for a, b, c in inst["triangles"]:
        out.append(transform_points(pts[np.asarray([a, b, c], dtype=int)], m))

    return np.asarray(out, dtype=np.float64)


def rasterize_xz_silhouette(triangles: np.ndarray, mn: np.ndarray, mx: np.ndarray, pixels_long: int):
    """
    Project triangles onto X-Z and fill their union.

    The returned matrix has image rows top-to-bottom (high Z to low Z).
    """
    width_m = float(mx[0] - mn[0])
    height_m = float(mx[2] - mn[2])
    longest = max(width_m, height_m)
    px_per_m = pixels_long / longest

    width_px = max(100, int(math.ceil(width_m * px_per_m)) + 4)
    height_px = max(100, int(math.ceil(height_m * px_per_m)) + 4)

    # Add two pixels padding around the world bbox.
    min_x = float(mn[0] - 2.0 / px_per_m)
    max_z = float(mx[2] + 2.0 / px_per_m)

    def world_to_pixel(points_xz: np.ndarray) -> np.ndarray:
        x_px = np.rint((points_xz[:, 0] - min_x) * px_per_m).astype(np.int32)
        y_px = np.rint((max_z - points_xz[:, 1]) * px_per_m).astype(np.int32)
        return np.c_[x_px, y_px]

    material = np.zeros((height_px, width_px), dtype=np.uint8)

    # Project each world-space triangle [x,y,z] -> [x,z].
    polygons = []
    for tri in triangles:
        projected = tri[:, [0, 2]]
        poly = world_to_pixel(projected).reshape((-1, 1, 2))
        polygons.append(poly)

    cv2.fillPoly(material, polygons, color=255, lineType=cv2.LINE_8)

    return material, px_per_m, min_x, max_z


def enclosed_openings(material: np.ndarray):
    """
    Opening pixels = free pixels not connected to outer image boundary.
    """
    free = (material == 0).astype(np.uint8)
    h, w = free.shape

    flood = free.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Start flood fill from every frame edge that is free.
    # Invert afterwards: free but not external means an enclosed opening.
    for x in range(w):
        if flood[0, x]:
            cv2.floodFill(flood, mask, (x, 0), 2)
        if flood[h - 1, x]:
            cv2.floodFill(flood, mask, (x, h - 1), 2)

    for y in range(h):
        if flood[y, 0]:
            cv2.floodFill(flood, mask, (0, y), 2)
        if flood[y, w - 1]:
            cv2.floodFill(flood, mask, (w - 1, y), 2)

    exterior = (flood == 2)
    opening = (free == 1) & (~exterior)

    # Open/close one pixel to remove isolated triangle seam specks.
    opening_u8 = opening.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    opening_u8 = cv2.morphologyEx(opening_u8, cv2.MORPH_OPEN, kernel)
    opening_u8 = cv2.morphologyEx(opening_u8, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(opening_u8, connectivity=8)
    return opening_u8, n, labels, stats, centroids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--px4-root", default=str(Path.home() / "PX4-Autopilot"))
    ap.add_argument("--out", default="scene_inspection_openings_fast")
    ap.add_argument(
        "--pixels-long",
        type=int,
        default=1200,
        help="Raster resolution along the larger left-frame dimension (default: 1200)",
    )
    args = ap.parse_args()

    px4_root = Path(args.px4_root).expanduser().resolve()
    dae = px4_root / "Tools/simulation/gz/models/test_world/meshes/test_terrain.dae"
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not dae.exists():
        print(f"ERROR: Missing DAE mesh: {dae}", file=sys.stderr)
        return 2

    root = ET.parse(dae).getroot()
    geometries = load_geometry(root)
    instances = gather_instances(root, geometries)
    left = choose_left_frame(instances)

    mn, mx = left["bounds_min"], left["bounds_max"]
    triangles = component_world_triangles(left)

    material, px_per_m, min_x, max_z = rasterize_xz_silhouette(
        triangles, mn, mx, max(400, args.pixels_long)
    )
    opening_u8, count, labels, stats, centroids = enclosed_openings(material)

    # Result image:
    # black material, white exterior, green detected opening.
    preview = np.full((material.shape[0], material.shape[1], 3), 255, dtype=np.uint8)
    preview[material > 0] = (30, 30, 30)
    preview[opening_u8 > 0] = (70, 210, 70)
    cv2.imwrite(str(out / "left_frame_opening_map.png"), preview)

    rows = []
    # label 0 is background
    for label_id in range(1, count):
        x_px, y_px, w_px, h_px, area = stats[label_id]
        if area < 25:
            continue

        # Pixel x increases with world X.
        # Pixel y increases toward lower world Z.
        low_x = min_x + x_px / px_per_m
        high_x = min_x + (x_px + w_px) / px_per_m
        high_z = max_z - y_px / px_per_m
        low_z = max_z - (y_px + h_px) / px_per_m

        width = high_x - low_x
        height = high_z - low_z

        rows.append({
            "opening_id": len(rows) + 1,
            "pixel_area": int(area),
            "min_x": float(low_x),
            "max_x": float(high_x),
            "min_z": float(low_z),
            "max_z": float(high_z),
            "width_x": float(width),
            "height_z": float(height),
            "center_x": float((low_x + high_x) / 2),
            "center_y": float((mn[1] + mx[1]) / 2),
            "center_z": float((low_z + high_z) / 2),
            "frame_min_y": float(mn[1]),
            "frame_max_y": float(mx[1]),
            "pixels_per_meter": float(px_per_m),
        })

    rows.sort(key=lambda r: r["width_x"] * r["height_z"], reverse=True)

    with (out / "left_frame_openings.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else [
            "opening_id", "pixel_area", "min_x", "max_x", "min_z", "max_z",
            "width_x", "height_z", "center_x", "center_y", "center_z",
            "frame_min_y", "frame_max_y", "pixels_per_meter",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (out / "left_frame_openings.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Left Frame Internal Opening Extraction — Fast Raster Method",
        "",
        f"Source DAE: `{dae}`",
        "",
        "## Selected left-frame outer bounds",
        f"- X: {mn[0]:.6f} to {mx[0]:.6f} m",
        f"- Y: {mn[1]:.6f} to {mx[1]:.6f} m",
        f"- Z: {mn[2]:.6f} to {mx[2]:.6f} m",
        f"- Raster resolution: {px_per_m:.1f} pixels/m",
        "",
        "## Detected enclosed openings",
        "",
        "| rank | width X (m) | height Z (m) | center X | center Y | center Z | min Z | max Z |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    if rows:
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row['width_x']:.4f} | {row['height_z']:.4f} | "
                f"{row['center_x']:.4f} | {row['center_y']:.4f} | {row['center_z']:.4f} | "
                f"{row['min_z']:.4f} | {row['max_z']:.4f} |"
            )
    else:
        lines.append("| - | no enclosed opening detected | | | | | | |")

    lines += [
        "",
        "## Files",
        "- `left_frame_opening_map.png`: black = material, green = detected opening",
        "- `left_frame_openings.csv`: numeric data",
        "- `left_frame_openings.json`: numeric data",
        "",
        "This extraction used the visual mesh silhouette only. Collision geometry",
        "and actual UAV clearance must still be verified before any traversal.",
    ]
    (out / "left_frame_openings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "README.txt").write_text(
        f"Done.\n"
        f"DAE: {dae}\n"
        f"Left frame outer bounds: X[{mn[0]:.3f}, {mx[0]:.3f}], "
        f"Y[{mn[1]:.3f}, {mx[1]:.3f}], Z[{mn[2]:.3f}, {mx[2]:.3f}]\n"
        f"Transformed triangles rasterized: {len(triangles)}\n"
        f"Detected enclosed openings: {len(rows)}\n"
        f"Output: {out}\n",
        encoding="utf-8",
    )

    print((out / "README.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
