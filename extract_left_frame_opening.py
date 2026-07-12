#!/usr/bin/env python3
"""
Read-only left-frame opening extractor for PX4 Gazebo test_world.

Goal
----
Extract the *internal opening(s)* of the left frame directly from the
test_terrain.dae mesh, instead of estimating the opening from the outer
bounding box.

Method
------
1. Parse test_terrain.dae and apply its visual-scene transforms.
2. Split mesh triangles into connected physical components.
3. Select the left frame component automatically from its known world bounds:
       x about -0.06 .. 1.14
       y about  3.08 .. 3.38
       z about  0.00 .. 1.85
4. Cast rays through the frame along Y on an X-Z grid.
5. Treat points inside material as occupied.
6. Find enclosed free-space regions: these are frame openings.
7. Report opening bounds / center / width / height.

Safety
------
This script is read-only.
It does NOT publish ROS 2 messages, command PX4, arm, take off, land,
or modify Gazebo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


# --------------------------- XML helpers ---------------------------

def lname(e: ET.Element) -> str:
    return e.tag.rsplit("}", 1)[-1]


def direct(parent: ET.Element | None, name: str) -> ET.Element | None:
    if parent is None:
        return None
    for child in list(parent):
        if lname(child) == name:
            return child
    return None


def fnums(text: str | None) -> list[float]:
    return [float(x) for x in (text or "").replace("\n", " ").split()]


def inums(text: str | None) -> list[int]:
    return [int(x) for x in (text or "").replace("\n", " ").split()]


# --------------------------- transforms ---------------------------

def translation(v: list[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[:3, 3] = v[:3]
    return m


def scale(v: list[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[0, 0], m[1, 1], m[2, 2] = v[:3]
    return m


def axis_angle(v: list[float]) -> np.ndarray:
    if len(v) < 4:
        return np.eye(4)
    axis = np.asarray(v[:3], dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(4)
    x, y, z = axis / norm
    a = math.radians(v[3])
    c, s = math.cos(a), math.sin(a)
    q = 1.0 - c
    r = np.array([
        [c + x*x*q,     x*y*q - z*s, x*z*q + y*s],
        [y*x*q + z*s,   c + y*y*q,   y*z*q - x*s],
        [z*x*q - y*s,   z*y*q + x*s, c + z*z*q],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    return m


def node_transform(node: ET.Element) -> np.ndarray:
    m = np.eye(4)
    for child in list(node):
        name = lname(child)
        vals = fnums(child.text)
        if name == "matrix" and len(vals) >= 16:
            local = np.asarray(vals[:16], dtype=float).reshape((4, 4), order="F")
        elif name == "translate":
            local = translation(vals)
        elif name == "scale":
            local = scale(vals)
        elif name == "rotate":
            local = axis_angle(vals)
        else:
            continue
        m = m @ local
    return m


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))]
    return (m @ homogeneous.T).T[:, :3]


# --------------------------- mesh parsing ---------------------------

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def read_sources(mesh: ET.Element) -> tuple[dict[str, np.ndarray], str | None]:
    sources: dict[str, np.ndarray] = {}

    for src in list(mesh):
        if lname(src) != "source":
            continue

        sid = src.attrib.get("id", "")
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
        sources[sid] = values.reshape((-1, stride))[:, :3]

    vertices = direct(mesh, "vertices")
    position_source = None

    if vertices is not None:
        for inp in list(vertices):
            if lname(inp) == "input" and inp.attrib.get("semantic") == "POSITION":
                position_source = inp.attrib.get("source", "").lstrip("#")
                break

    return sources, position_source


def triangles_for_primitive(prim: ET.Element) -> list[tuple[int, int, int]]:
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
    for child in list(prim):
        if lname(child) == "p":
            raw.extend(inums(child.text))

    primitive_name = lname(prim)

    if primitive_name == "triangles":
        count = int(prim.attrib.get("count", "0"))
        raw = raw[:count * 3 * stride]
        output = []

        for i in range(0, len(raw), 3 * stride):
            if i + 3 * stride <= len(raw):
                output.append((
                    raw[i + vertex_offset],
                    raw[i + stride + vertex_offset],
                    raw[i + 2 * stride + vertex_offset],
                ))
        return output

    if primitive_name == "polylist":
        vcount = direct(prim, "vcount")
        if vcount is None:
            return []

        cursor = 0
        output = []

        for count in inums(vcount.text):
            polygon = [
                raw[cursor + j * stride + vertex_offset]
                for j in range(count)
                if cursor + j * stride + vertex_offset < len(raw)
            ]
            cursor += count * stride

            for j in range(1, len(polygon) - 1):
                output.append((polygon[0], polygon[j], polygon[j + 1]))

        return output

    return []


def geometry_data(root: ET.Element) -> dict[str, dict]:
    """Return local positions and triangles per geometry id."""
    output: dict[str, dict] = {}

    for geo in root.iter():
        if lname(geo) != "geometry":
            continue

        mesh = direct(geo, "mesh")
        if mesh is None:
            continue

        sources, pos_id = read_sources(mesh)

        if not pos_id or pos_id not in sources:
            continue

        triangles = []

        for primitive in list(mesh):
            if lname(primitive) in ("triangles", "polylist"):
                triangles.extend(triangles_for_primitive(primitive))

        if not triangles:
            continue

        output[geo.attrib.get("id", "")] = {
            "name": geo.attrib.get("name", geo.attrib.get("id", "")),
            "positions": sources[pos_id],
            "triangles": triangles,
        }

    return output


def component_vertex_groups(position_count: int,
                            triangles: list[tuple[int, int, int]]) -> list[set[int]]:
    valid = [
        t for t in triangles
        if all(0 <= idx < position_count for idx in t)
    ]
    used = sorted({idx for tri in valid for idx in tri})

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


def component_triangles(triangles: list[tuple[int, int, int]],
                        group: set[int]) -> list[tuple[int, int, int]]:
    return [
        tri for tri in triangles
        if tri[0] in group and tri[1] in group and tri[2] in group
    ]


def world_instances(root: ET.Element, geos: dict[str, dict]) -> list[dict]:
    visual_scene = next(
        (e for e in root.iter() if lname(e) == "visual_scene"),
        None,
    )

    if visual_scene is None:
        return []

    instances = []

    def walk(node: ET.Element, parent_m: np.ndarray, path: list[str]) -> None:
        node_name = node.attrib.get("name", node.attrib.get("id", "node"))
        m = parent_m @ node_transform(node)
        current_path = path + [node_name]

        for child in list(node):
            if lname(child) == "instance_geometry":
                gid = child.attrib.get("url", "").lstrip("#")

                if gid not in geos:
                    continue

                data = geos[gid]
                positions = data["positions"]
                triangles = data["triangles"]

                for comp_index, group in enumerate(
                    component_vertex_groups(len(positions), triangles),
                    start=1,
                ):
                    ids = np.asarray(sorted(group), dtype=int)
                    world_pts = transform_points(positions[ids], m)
                    tri = component_triangles(triangles, group)

                    if not tri:
                        continue

                    mn = world_pts.min(axis=0)
                    mx = world_pts.max(axis=0)

                    instances.append({
                        "node_path": "/".join(current_path),
                        "geometry_name": data["name"],
                        "component_index": comp_index,
                        "local_positions": positions,
                        "world_matrix": m,
                        "group": group,
                        "triangles": tri,
                        "min": mn,
                        "max": mx,
                    })

            elif lname(child) == "node":
                walk(child, m, current_path)

    for child in list(visual_scene):
        if lname(child) == "node":
            walk(child, np.eye(4), [])

    return instances


# --------------------------- opening extraction ---------------------------

def choose_left_frame(instances: list[dict]) -> dict:
    """
    Pick the tall left-frame component:
      approximately X [-0.06, 1.14], Y [3.08, 3.38], Z [0, 1.85].
    """
    best = None
    best_score = float("inf")

    target = np.array([
        -0.06, 1.14,
         3.078, 3.378,
         0.0, 1.853,
    ])

    for inst in instances:
        mn, mx = inst["min"], inst["max"]
        values = np.array([mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]])
        score = float(np.linalg.norm(values - target))

        if score < best_score:
            best_score = score
            best = inst

    if best is None:
        raise RuntimeError("Could not identify the left frame component.")

    return best


def world_component_triangles(instance: dict) -> np.ndarray:
    """
    Return triangles as shape (N, 3, 3) in world coordinates.
    """
    positions = instance["local_positions"]
    m = instance["world_matrix"]

    triangles = []

    for a, b, c in instance["triangles"]:
        pts = transform_points(
            positions[np.asarray([a, b, c], dtype=int)],
            m,
        )
        triangles.append(pts)

    return np.asarray(triangles, dtype=float)


def ray_intersects_triangle_y(
    x: float,
    z: float,
    y0: float,
    triangle: np.ndarray,
    eps: float = 1e-9,
) -> float | None:
    """
    Intersect a ray p(y) = (x, y0+t, z), t>=0 with a triangle.

    Returns world Y coordinate of intersection, or None.
    """
    p = np.array([x, y0, z], dtype=float)
    d = np.array([0.0, 1.0, 0.0], dtype=float)

    v0, v1, v2 = triangle
    e1 = v1 - v0
    e2 = v2 - v0

    h = np.cross(d, e2)
    a = np.dot(e1, h)

    if abs(a) < eps:
        return None

    f = 1.0 / a
    s = p - v0
    u = f * np.dot(s, h)

    if u < -eps or u > 1.0 + eps:
        return None

    q = np.cross(s, e1)
    v = f * np.dot(d, q)

    if v < -eps or u + v > 1.0 + eps:
        return None

    t = f * np.dot(e2, q)

    if t <= eps:
        return None

    return y0 + t


def point_inside_frame_material(
    x: float,
    z: float,
    triangles: np.ndarray,
    y_start: float,
) -> bool:
    """
    Odd-even ray test against the closed frame mesh.
    Deduplicates almost-equal intersections at shared triangle edges.
    """
    hits = []

    for tri in triangles:
        y_hit = ray_intersects_triangle_y(x, z, y_start, tri)
        if y_hit is not None:
            hits.append(y_hit)

    if not hits:
        return False

    hits.sort()

    unique = []
    tol = 1e-6

    for value in hits:
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)

    return len(unique) % 2 == 1


def external_air_mask(free_mask: np.ndarray) -> np.ndarray:
    """
    free_mask True = non-material.
    Return free cells connected to grid boundary.
    """
    h, w = free_mask.shape
    seen = np.zeros_like(free_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int):
        if free_mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))

    for c in range(w):
        add(0, c)
        add(h - 1, c)

    for r in range(h):
        add(r, 0)
        add(r, w - 1)

    while q:
        r, c = q.popleft()

        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w:
                add(rr, cc)

    return seen


def connected_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Find connected True regions with 4-neighbour connectivity.
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            region = []

            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))

                for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        if mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))

            out.append(region)

    return out


def write_ascii_map(path: Path, material: np.ndarray, opening: np.ndarray) -> None:
    """
    Top of text file is high Z. '#' material, '.' external air, 'O' opening.
    """
    h, w = material.shape
    lines = [
        "# Left-frame X-Z occupancy map",
        "# Top line = max Z, left = min X",
        "# #: frame material, .: external air, O: enclosed opening",
        "",
    ]

    for r in range(h - 1, -1, -1):
        chars = []

        for c in range(w):
            if opening[r, c]:
                chars.append("O")
            elif material[r, c]:
                chars.append("#")
            else:
                chars.append(".")

        lines.append("".join(chars))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--px4-root",
        default=str(Path.home() / "PX4-Autopilot"),
    )
    parser.add_argument(
        "--out",
        default="scene_inspection_openings",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=180,
        help="Grid cells along the longer X/Z frame dimension.",
    )
    args = parser.parse_args()

    px4_root = Path(args.px4_root).expanduser().resolve()
    dae_path = (
        px4_root
        / "Tools/simulation/gz/models/test_world/meshes/test_terrain.dae"
    )
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dae_path.exists():
        print(f"ERROR: file not found: {dae_path}", file=sys.stderr)
        return 2

    root = ET.parse(dae_path).getroot()
    geos = geometry_data(root)
    instances = world_instances(root, geos)
    left = choose_left_frame(instances)
    triangles = world_component_triangles(left)

    mn, mx = left["min"], left["max"]

    # Slightly shrink from outer boundary so a ray exactly at the mesh edge
    # does not create numerical ambiguities.
    margin_x = max(1e-4, (mx[0] - mn[0]) * 0.001)
    margin_z = max(1e-4, (mx[2] - mn[2]) * 0.001)

    x0, x1 = mn[0] + margin_x, mx[0] - margin_x
    z0, z1 = mn[2] + margin_z, mx[2] - margin_z

    longer = max(x1 - x0, z1 - z0)
    cells_long = max(50, int(args.resolution))
    dx = longer / cells_long
    nx = max(30, int(math.ceil((x1 - x0) / dx)))
    nz = max(30, int(math.ceil((z1 - z0) / dx)))

    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(z0, z1, nz)
    y_start = mn[1] - 0.05

    material = np.zeros((nz, nx), dtype=bool)

    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            material[iz, ix] = point_inside_frame_material(
                float(x),
                float(z),
                triangles,
                float(y_start),
            )

    free = ~material
    exterior = external_air_mask(free)
    opening_mask = free & ~exterior
    regions = connected_regions(opening_mask)

    # Convert regions into world-space bounds.
    opening_rows = []

    for idx, region in enumerate(regions, start=1):
        if len(region) < 4:
            continue

        rs = [r for r, _ in region]
        cs = [c for _, c in region]

        min_x = float(xs[min(cs)])
        max_x = float(xs[max(cs)])
        min_z = float(zs[min(rs)])
        max_z = float(zs[max(rs)])

        opening_rows.append({
            "opening_id": idx,
            "cell_count": len(region),
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
            "width_x": max_x - min_x,
            "height_z": max_z - min_z,
            "center_x": (min_x + max_x) / 2.0,
            "center_y": float((mn[1] + mx[1]) / 2.0),
            "center_z": (min_z + max_z) / 2.0,
            "frame_min_y": float(mn[1]),
            "frame_max_y": float(mx[1]),
            "grid_nx": nx,
            "grid_nz": nz,
        })

    opening_rows.sort(
        key=lambda row: row["width_x"] * row["height_z"],
        reverse=True,
    )

    with (out_dir / "left_frame_openings.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = list(opening_rows[0].keys()) if opening_rows else [
            "opening_id", "cell_count",
            "min_x", "max_x", "min_z", "max_z",
            "width_x", "height_z",
            "center_x", "center_y", "center_z",
            "frame_min_y", "frame_max_y",
            "grid_nx", "grid_nz",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(opening_rows)

    (out_dir / "left_frame_openings.json").write_text(
        json.dumps(opening_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_ascii_map(
        out_dir / "left_frame_xz_occupancy.txt",
        material,
        opening_mask,
    )

    lines = [
        "# Left Frame Internal Opening Extraction",
        "",
        "Source mesh: " + str(dae_path),
        "",
        "Selected left-frame outer bounds:",
        f"- X: {mn[0]:.6f} to {mx[0]:.6f} m",
        f"- Y: {mn[1]:.6f} to {mx[1]:.6f} m",
        f"- Z: {mn[2]:.6f} to {mx[2]:.6f} m",
        "",
        "The Y coordinate below is the center plane of the frame thickness.",
        "Opening measurements are derived from enclosed free regions in an X-Z",
        "occupancy slice through the frame.",
        "",
        "## Detected enclosed openings",
        "",
        "| rank | opening | width X | height Z | center X | center Y | center Z | min Z | max Z |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    if opening_rows:
        for rank, row in enumerate(opening_rows, start=1):
            lines.append(
                f"| {rank} | {row['opening_id']} | "
                f"{row['width_x']:.4f} | {row['height_z']:.4f} | "
                f"{row['center_x']:.4f} | {row['center_y']:.4f} | "
                f"{row['center_z']:.4f} | {row['min_z']:.4f} | "
                f"{row['max_z']:.4f} |"
            )
    else:
        lines.append(
            "| - | - | No enclosed opening was detected. See occupancy text map. |"
        )

    lines.extend([
        "",
        "## Important interpretation",
        "",
        "- These are geometric opening estimates from the visual mesh.",
        "- Before any flight, collision geometry and drone body clearance still",
        "  need to be checked separately.",
        "- The script did not control or move the UAV.",
        "",
        "Files:",
        "- `left_frame_openings.csv`: numeric results",
        "- `left_frame_xz_occupancy.txt`: X-Z text map (# material, O opening)",
    ])

    (out_dir / "left_frame_openings.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "Done.\n"
        f"Selected left frame: X[{mn[0]:.3f}, {mx[0]:.3f}], "
        f"Y[{mn[1]:.3f}, {mx[1]:.3f}], "
        f"Z[{mn[2]:.3f}, {mx[2]:.3f}]\n"
        f"Grid: {nx} x {nz}\n"
        f"Detected enclosed openings: {len(opening_rows)}\n"
        f"Output: {out_dir}\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
