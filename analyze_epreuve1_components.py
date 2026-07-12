#!/usr/bin/env python3
"""
Split the merged `epreuve_1` COLLADA mesh into connected components.

Why this exists:
`analyze_test_world_geometry.py` showed that epreuve_1 is one large merged
geometry. Its full bounding box includes both empty ground and obstacles, so it
cannot be used directly as a no-fly rectangle. This script rebuilds the mesh
connectivity from triangle indices and reports each disconnected component.

Output:
  scene_analysis/epreuve_1_components.csv
  scene_analysis/epreuve_1_components.json

Safety note:
This script maps geometry only. A component with vertical height is a candidate
obstacle, but safe route planning still requires a coordinate calibration from
Gazebo world coordinates to PX4 local NED coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
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


def ints(text: Optional[str]) -> List[int]:
    return [int(x) for x in text.split()] if text else []


def strip_ref(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value[1:] if value.startswith("#") else value


def identity() -> List[float]:
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def transform_row(p: Vec3, m: Sequence[float]) -> Vec3:
    x, y, z = p
    tx = m[0]*x + m[1]*y + m[2]*z + m[3]
    ty = m[4]*x + m[5]*y + m[6]*z + m[7]
    tz = m[8]*x + m[9]*y + m[10]*z + m[11]
    tw = m[12]*x + m[13]*y + m[14]*z + m[15]
    if abs(tw) > 1e-12 and abs(tw - 1.0) > 1e-12:
        tx, ty, tz = tx/tw, ty/tw, tz/tw
    return tx, ty, tz


def bounds(points: Sequence[Vec3]) -> Bounds:
    xs, ys, zs = zip(*points)
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def bounds_str(b: Bounds) -> str:
    return (
        f"x=[{b[0]:.3f}, {b[1]:.3f}], "
        f"y=[{b[2]:.3f}, {b[3]:.3f}], "
        f"z=[{b[4]:.3f}, {b[5]:.3f}]"
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def add(self, value: int) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: int) -> int:
        self.add(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def geometry_by_id(root: ET.Element, geometry_id: str) -> ET.Element:
    for item in root.iter():
        if lname(item.tag) == "geometry" and item.attrib.get("id") == geometry_id:
            return item
    raise ValueError(f"Geometry not found: {geometry_id}")


def source_points(mesh: ET.Element) -> Tuple[List[Vec3], Dict[str, int]]:
    """
    Return POSITION points and mapping source-id -> point stride.
    """
    vertices = child(mesh, "vertices")
    if vertices is None:
        raise ValueError("mesh has no <vertices>")

    position_source_id = None
    for inp in children(vertices, "input"):
        if inp.attrib.get("semantic") == "POSITION":
            position_source_id = strip_ref(inp.attrib.get("source"))
            break
    if not position_source_id:
        raise ValueError("<vertices> has no POSITION input")

    position_source = None
    for source in children(mesh, "source"):
        if source.attrib.get("id") == position_source_id:
            position_source = source
            break
    if position_source is None:
        raise ValueError("POSITION source was not found")

    array = child(position_source, "float_array")
    if array is None:
        raise ValueError("POSITION source has no float_array")

    values = nums(array.text)
    stride = 3
    tc = child(position_source, "technique_common")
    acc = child(tc, "accessor") if tc is not None else None
    if acc is not None:
        stride = int(acc.attrib.get("stride", "3"))

    if stride < 3:
        raise ValueError("POSITION stride is smaller than 3")

    points = [
        (values[i], values[i+1], values[i+2])
        for i in range(0, len(values) - stride + 1, stride)
    ]
    return points, {position_source_id: stride}


def vertices_position_source(mesh: ET.Element) -> str:
    verts = child(mesh, "vertices")
    if verts is None:
        raise ValueError("mesh has no <vertices>")
    for inp in children(verts, "input"):
        if inp.attrib.get("semantic") == "POSITION":
            return strip_ref(inp.attrib.get("source")) or ""
    raise ValueError("no POSITION source under <vertices>")


def primitive_vertex_indices(mesh: ET.Element) -> List[Tuple[int, int, int]]:
    """
    Extract triangle vertex indices from <triangles> and <polylist>.
    Only VERTEX/POSITION stream index is needed for connectivity.
    """
    vertex_source_id = vertices_position_source(mesh)
    triangles: List[Tuple[int, int, int]] = []

    for primitive in mesh:
        primitive_type = lname(primitive.tag)
        if primitive_type not in ("triangles", "polylist", "polygons"):
            continue

        inputs = []
        max_offset = 0
        vertex_offset = None
        for inp in children(primitive, "input"):
            offset = int(inp.attrib.get("offset", "0"))
            semantic = inp.attrib.get("semantic", "")
            source = strip_ref(inp.attrib.get("source"))
            inputs.append((semantic, source, offset))
            max_offset = max(max_offset, offset)

            # VERTEX points at <vertices>; that element maps to POSITION.
            if semantic == "VERTEX":
                vertex_offset = offset
            elif semantic == "POSITION" and source == vertex_source_id:
                vertex_offset = offset

        if vertex_offset is None:
            continue

        tuple_width = max_offset + 1
        p_nodes = list(children(primitive, "p"))
        raw = []
        for p in p_nodes:
            raw.extend(ints(p.text))

        if primitive_type == "triangles":
            count = int(primitive.attrib.get("count", "0"))
            expected = count * 3 * tuple_width
            if expected and len(raw) < expected:
                print(
                    f"Warning: triangle index data shorter than expected "
                    f"({len(raw)} < {expected})",
                    file=sys.stderr,
                )
            usable = min(len(raw), expected) if expected else len(raw)
            for i in range(0, usable - 3 * tuple_width + 1, 3 * tuple_width):
                a = raw[i + vertex_offset]
                b = raw[i + tuple_width + vertex_offset]
                c = raw[i + 2 * tuple_width + vertex_offset]
                triangles.append((a, b, c))

        elif primitive_type == "polylist":
            vcount = child(primitive, "vcount")
            if vcount is None:
                continue
            polygon_sizes = ints(vcount.text)
            cursor = 0
            for nverts in polygon_sizes:
                polygon = []
                for _ in range(nverts):
                    if cursor + tuple_width > len(raw):
                        break
                    polygon.append(raw[cursor + vertex_offset])
                    cursor += tuple_width
                # Fan triangulation for connectivity.
                for idx in range(1, len(polygon) - 1):
                    triangles.append((polygon[0], polygon[idx], polygon[idx + 1]))

        elif primitive_type == "polygons":
            for p in p_nodes:
                poly_raw = ints(p.text)
                polygon = [
                    poly_raw[i + vertex_offset]
                    for i in range(0, len(poly_raw) - tuple_width + 1, tuple_width)
                ]
                for idx in range(1, len(polygon) - 1):
                    triangles.append((polygon[0], polygon[idx], polygon[idx + 1]))

    return triangles


def geometry_matrix(root: ET.Element, geometry_id: str) -> List[float]:
    for node in root.iter():
        if lname(node.tag) != "node":
            continue
        for inst in node.iter():
            if lname(inst.tag) != "instance_geometry":
                continue
            if strip_ref(inst.attrib.get("url")) == geometry_id:
                matrix_node = child(node, "matrix")
                matrix = nums(matrix_node.text) if matrix_node is not None else identity()
                return matrix if len(matrix) == 16 else identity()
    return identity()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dae",
        nargs="?",
        default="~/PX4-Autopilot/Tools/simulation/gz/models/test_world/meshes/test_terrain.dae",
    )
    parser.add_argument("--geometry", default="epreuve_1-mesh")
    parser.add_argument(
        "--out",
        default="~/PX4-ROS2-Gazebo-Drone-Simulation-Template/scene_analysis",
    )
    parser.add_argument(
        "--min-triangles",
        type=int,
        default=8,
        help="Hide tiny disconnected fragments below this triangle count.",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.10,
        help="Mark components higher than this as vertical obstacle candidates.",
    )
    args = parser.parse_args()

    dae = Path(args.dae).expanduser()
    out = Path(args.out).expanduser()
    if not dae.is_file():
        print(f"ERROR: DAE file not found: {dae}", file=sys.stderr)
        return 2

    root = ET.parse(dae).getroot()
    geometry = geometry_by_id(root, args.geometry)
    mesh = child(geometry, "mesh")
    if mesh is None:
        print("ERROR: selected geometry has no mesh", file=sys.stderr)
        return 3

    points, _ = source_points(mesh)
    triangles = primitive_vertex_indices(mesh)
    if not triangles:
        print("ERROR: no triangle or polygon indices extracted", file=sys.stderr)
        return 4

    uf = UnionFind()
    used_vertices = set()
    for a, b, c in triangles:
        uf.union(a, b)
        uf.union(a, c)
        used_vertices.update((a, b, c))

    groups: Dict[int, set[int]] = defaultdict(set)
    triangle_counts: Dict[int, int] = defaultdict(int)

    for vertex in used_vertices:
        groups[uf.find(vertex)].add(vertex)

    for a, _, _ in triangles:
        triangle_counts[uf.find(a)] += 1

    matrix = geometry_matrix(root, args.geometry)
    rows = []
    for root_id, vertex_set in groups.items():
        count = triangle_counts[root_id]
        if count < args.min_triangles:
            continue

        valid = [i for i in vertex_set if 0 <= i < len(points)]
        if not valid:
            continue

        transformed = [transform_row(points[i], matrix) for i in valid]
        b = bounds(transformed)
        width = b[1] - b[0]
        depth = b[3] - b[2]
        height = b[5] - b[4]
        candidate = height >= args.min_height

        rows.append(
            {
                "component_id": 0,  # assigned after sorting
                "triangle_count": count,
                "vertex_count": len(valid),
                "x_min": b[0],
                "x_max": b[1],
                "y_min": b[2],
                "y_max": b[3],
                "z_min": b[4],
                "z_max": b[5],
                "width": width,
                "depth": depth,
                "height": height,
                "vertical_obstacle_candidate": candidate,
            }
        )

    # Largest/most complex components first.
    rows.sort(key=lambda r: (r["vertical_obstacle_candidate"], r["triangle_count"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["component_id"] = index

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "epreuve_1_components.csv"
    json_path = out / "epreuve_1_components.json"

    fields = [
        "component_id",
        "triangle_count",
        "vertex_count",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "width",
        "depth",
        "height",
        "vertical_obstacle_candidate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== epreuve_1 connected components ===")
    print(f"Geometry: {args.geometry}")
    print(f"Vertices: {len(points)}")
    print(f"Triangles: {len(triangles)}")
    print(f"Components shown: {len(rows)}")
    print(
        f"Obstacle-candidate rule: height >= {args.min_height:.2f} m "
        f"and triangle_count >= {args.min_triangles}"
    )
    print()

    for row in rows:
        marker = "OBSTACLE?" if row["vertical_obstacle_candidate"] else "flat/small"
        print(
            f"C{row['component_id']:02d} [{marker}] "
            f"tri={row['triangle_count']}, vertices={row['vertex_count']} | "
            f"x=[{row['x_min']:.3f}, {row['x_max']:.3f}], "
            f"y=[{row['y_min']:.3f}, {row['y_max']:.3f}], "
            f"z=[{row['z_min']:.3f}, {row['z_max']:.3f}] | "
            f"W={row['width']:.3f}, D={row['depth']:.3f}, H={row['height']:.3f}"
        )

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
