#!/usr/bin/env python3
"""
Extract geometry bounds from the PX4 Gazebo test_world COLLADA (.dae) file.

This is a map-extraction helper. It does not determine flight safety by itself.
It writes CSV and JSON reports with raw, row-major, and column-major bounds
for each named geometry so we can later align the terrain model with PX4 NED.
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Bounds = Tuple[float, float, float, float, float, float]


def lname(tag: str) -> str:
    return tag.split("}", 1)[-1]


def child(parent: ET.Element, name: str) -> Optional[ET.Element]:
    for item in parent:
        if lname(item.tag) == name:
            return item
    return None


def children(parent: ET.Element, name: str) -> Iterable[ET.Element]:
    for item in parent:
        if lname(item.tag) == name:
            yield item


def nums(text: Optional[str]) -> List[float]:
    return [float(x) for x in text.split()] if text else []


def strip_ref(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value[1:] if value.startswith("#") else value


def identity() -> List[float]:
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def bounds(points: Sequence[Vec3]) -> Bounds:
    xs, ys, zs = zip(*points)
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def show(b: Bounds) -> str:
    return (
        f"x=[{b[0]:.3f}, {b[1]:.3f}], "
        f"y=[{b[2]:.3f}, {b[3]:.3f}], "
        f"z=[{b[4]:.3f}, {b[5]:.3f}]"
    )


def point_row(p: Vec3, m: Sequence[float]) -> Vec3:
    x, y, z = p
    tx = m[0]*x + m[1]*y + m[2]*z + m[3]
    ty = m[4]*x + m[5]*y + m[6]*z + m[7]
    tz = m[8]*x + m[9]*y + m[10]*z + m[11]
    tw = m[12]*x + m[13]*y + m[14]*z + m[15]
    if abs(tw) > 1e-12 and abs(tw - 1.0) > 1e-12:
        tx, ty, tz = tx/tw, ty/tw, tz/tw
    return tx, ty, tz


def point_col(p: Vec3, m: Sequence[float]) -> Vec3:
    x, y, z = p
    tx = m[0]*x + m[4]*y + m[8]*z + m[12]
    ty = m[1]*x + m[5]*y + m[9]*z + m[13]
    tz = m[2]*x + m[6]*y + m[10]*z + m[14]
    tw = m[3]*x + m[7]*y + m[11]*z + m[15]
    if abs(tw) > 1e-12 and abs(tw - 1.0) > 1e-12:
        tx, ty, tz = tx/tw, ty/tw, tz/tw
    return tx, ty, tz


def geometry_points(geometry: ET.Element) -> List[Vec3]:
    mesh = child(geometry, "mesh")
    if mesh is None:
        return []

    verts = child(mesh, "vertices")
    if verts is None:
        return []

    source_id = None
    for inp in children(verts, "input"):
        if inp.attrib.get("semantic") == "POSITION":
            source_id = strip_ref(inp.attrib.get("source"))
            break
    if not source_id:
        return []

    source = None
    for item in children(mesh, "source"):
        if item.attrib.get("id") == source_id:
            source = item
            break
    if source is None:
        return []

    array = child(source, "float_array")
    if array is None:
        return []

    values = nums(array.text)
    stride = 3
    tc = child(source, "technique_common")
    accessor = child(tc, "accessor") if tc is not None else None
    if accessor is not None:
        try:
            stride = int(accessor.attrib.get("stride", "3"))
        except ValueError:
            stride = 3

    if stride < 3:
        return []

    return [
        (values[i], values[i+1], values[i+2])
        for i in range(0, len(values) - stride + 1, stride)
    ]


def geometry_to_node(root: ET.Element) -> Dict[str, Tuple[str, List[float]]]:
    mapping: Dict[str, Tuple[str, List[float]]] = {}
    for node in root.iter():
        if lname(node.tag) != "node":
            continue
        name = node.attrib.get("name") or node.attrib.get("id") or "unnamed"
        matrix_node = child(node, "matrix")
        matrix = nums(matrix_node.text) if matrix_node is not None else identity()
        if len(matrix) != 16:
            matrix = identity()

        for item in node.iter():
            if lname(item.tag) != "instance_geometry":
                continue
            gid = strip_ref(item.attrib.get("url"))
            if gid:
                mapping[gid] = (name, matrix)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dae",
        nargs="?",
        default="~/PX4-Autopilot/Tools/simulation/gz/models/test_world/meshes/test_terrain.dae",
    )
    parser.add_argument(
        "--out",
        default="~/PX4-ROS2-Gazebo-Drone-Simulation-Template/scene_analysis",
    )
    args = parser.parse_args()

    dae = Path(args.dae).expanduser()
    out = Path(args.out).expanduser()
    if not dae.is_file():
        print(f"ERROR: DAE not found: {dae}", file=sys.stderr)
        return 2

    root = ET.parse(dae).getroot()
    node_map = geometry_to_node(root)
    rows = []

    for geo in root.iter():
        if lname(geo.tag) != "geometry":
            continue
        pts = geometry_points(geo)
        if not pts:
            continue

        gid = geo.attrib.get("id", "unnamed_geometry")
        gname = geo.attrib.get("name", gid)
        node_name, matrix = node_map.get(gid, (gname, identity()))

        raw = bounds(pts)
        row = bounds([point_row(p, matrix) for p in pts])
        col = bounds([point_col(p, matrix) for p in pts])

        rows.append(
            {
                "geometry_id": gid,
                "geometry_name": gname,
                "scene_node": node_name,
                "vertex_count": len(pts),
                "raw_bounds": raw,
                "row_major_bounds": row,
                "column_major_bounds": col,
                "matrix": matrix,
            }
        )

    if not rows:
        print("ERROR: no POSITION vertex data found", file=sys.stderr)
        return 3

    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "test_world_geometry_bounds.json"
    csv_path = out / "test_world_geometry_bounds.csv"

    json_path.write_text(
        json.dumps(
            [
                {
                    **r,
                    "raw_bounds": list(r["raw_bounds"]),
                    "row_major_bounds": list(r["row_major_bounds"]),
                    "column_major_bounds": list(r["column_major_bounds"]),
                }
                for r in rows
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fields = [
        "geometry_id",
        "geometry_name",
        "scene_node",
        "vertex_count",
        "raw_bounds",
        "row_major_bounds",
        "column_major_bounds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "geometry_id": r["geometry_id"],
                    "geometry_name": r["geometry_name"],
                    "scene_node": r["scene_node"],
                    "vertex_count": r["vertex_count"],
                    "raw_bounds": show(r["raw_bounds"]),
                    "row_major_bounds": show(r["row_major_bounds"]),
                    "column_major_bounds": show(r["column_major_bounds"]),
                }
            )

    print("\n=== test_world geometry bounds ===")
    for r in rows:
        print(f"\n[{r['geometry_name']}]")
        print(f"  node       : {r['scene_node']}")
        print(f"  vertices   : {r['vertex_count']}")
        print(f"  raw        : {show(r['raw_bounds'])}")
        print(f"  row-major  : {show(r['row_major_bounds'])}")
        print(f"  col-major  : {show(r['column_major_bounds'])}")

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
