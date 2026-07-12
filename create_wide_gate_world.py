#!/usr/bin/env python3
"""
Create a new PX4/Gazebo world from the existing test_world, but replace ONLY
the narrow left frame (C01) with a wider, taller, collision-safe gate.

Original files are never modified.

Created:
  ~/PX4-Autopilot/Tools/simulation/gz/models/test_world_gate_wide/
  ~/PX4-Autopilot/Tools/simulation/gz/worlds/test_world_gate_wide.sdf

New gate geometry (same approximate original left-gate location):
  outer X: -0.060 .. 1.140 m
  outer Y:  3.078 .. 3.378 m
  outer Z:  0.000 .. 1.853 m

  clear opening X: 0.050 .. 1.030 m   -> 0.980 m width
  clear opening Z: 0.200 .. 1.680 m   -> 1.480 m height

The old left frame is removed from both the visual and collision terrain mesh.
The new gate is added as explicit SDF box collision/visual links. The rest of
the original terrain / obstacles remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from lxml import etree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Original left-frame world-space bounds from prior verified extraction.
TARGET_BOUNDS = np.array([
    -0.060367, 1.139633,
     3.077722, 3.377722,
     0.000000, 1.852962,
], dtype=float)

# Replacement gate's OPEN area — deliberately still gate-like, not toy-large.
OPEN_MIN_X, OPEN_MAX_X = 0.050, 1.030
OPEN_MIN_Z, OPEN_MAX_Z = 0.200, 1.680
GATE_MIN_X, GATE_MAX_X = -0.060367, 1.139633
GATE_MIN_Y, GATE_MAX_Y = 3.077722, 3.377722
GATE_MIN_Z, GATE_MAX_Z = 0.000000, 1.852962


def lname(e: ET.Element) -> str:
    return e.tag.rsplit("}", 1)[-1]


def direct(parent: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if parent is None:
        return None
    for child in list(parent):
        if lname(child) == name:
            return child
    return None


def fnums(text: Optional[str]) -> List[float]:
    return [float(x) for x in (text or "").replace("\n", " ").split()]


def inums(text: Optional[str]) -> List[int]:
    return [int(x) for x in (text or "").replace("\n", " ").split()]


def fmt_ints(values: Sequence[int], wrap: int = 48) -> str:
    tokens = [str(v) for v in values]
    return "\n".join(
        " ".join(tokens[i:i + wrap]) for i in range(0, len(tokens), wrap)
    )


def translation(v: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[:3, 3] = v[:3]
    return m


def scaling(v: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    if len(v) >= 3:
        m[0, 0], m[1, 1], m[2, 2] = v[:3]
    return m


def axis_angle(v: Sequence[float]) -> np.ndarray:
    if len(v) < 4:
        return np.eye(4)
    axis = np.asarray(v[:3], dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(4)
    x, y, z = axis / norm
    angle = np.deg2rad(v[3])
    c, s = np.cos(angle), np.sin(angle)
    q = 1.0 - c
    r = np.array([
        [c + x*x*q, x*y*q - z*s, x*z*q + y*s],
        [y*x*q + z*s, c + y*y*q, y*z*q - x*s],
        [z*x*q - y*s, z*y*q + x*s, c + z*z*q],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    return m


def node_transform(node: ET.Element) -> np.ndarray:
    m = np.eye(4)
    for child in list(node):
        name = lname(child)
        values = fnums(child.text)
        if name == "matrix" and len(values) >= 16:
            local = np.asarray(values[:16], dtype=float).reshape((4, 4), order="F")
        elif name == "translate":
            local = translation(values)
        elif name == "scale":
            local = scaling(values)
        elif name == "rotate":
            local = axis_angle(values)
        else:
            continue
        m = m @ local
    return m


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (m @ np.c_[points, np.ones(len(points))].T).T[:, :3]


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def read_mesh_positions(mesh: ET.Element) -> Tuple[Dict[str, np.ndarray], Optional[str]]:
    sources: Dict[str, np.ndarray] = {}
    for source in list(mesh):
        if lname(source) != "source":
            continue
        source_id = source.attrib.get("id", "")
        float_array = direct(source, "float_array")
        technique = direct(source, "technique_common")
        accessor = direct(technique, "accessor")
        if float_array is None:
            continue
        stride = int(accessor.attrib.get("stride", "3")) if accessor is not None else 3
        vals = np.asarray(fnums(float_array.text), dtype=float)
        if stride < 3 or vals.size < 3:
            continue
        vals = vals[:(vals.size // stride)*stride].reshape((-1, stride))[:, :3]
        sources[source_id] = vals

    vertices = direct(mesh, "vertices")
    pos_source = None
    if vertices is not None:
        for inp in list(vertices):
            if lname(inp) == "input" and inp.attrib.get("semantic") == "POSITION":
                pos_source = inp.attrib.get("source", "").lstrip("#")
                break
    return sources, pos_source


def primitive_layout(prim: ET.Element) -> Tuple[int, Optional[int], List[int]]:
    inputs = [c for c in list(prim) if lname(c) == "input"]
    if not inputs:
        return 0, None, []
    stride = max(int(i.attrib.get("offset", "0")) for i in inputs) + 1
    vertex_offset = None
    for inp in inputs:
        if inp.attrib.get("semantic") == "VERTEX":
            vertex_offset = int(inp.attrib.get("offset", "0"))
            break
    raw: List[int] = []
    for child in list(prim):
        if lname(child) == "p":
            raw.extend(inums(child.text))
    return stride, vertex_offset, raw


def extract_triangles(prim: ET.Element) -> List[Tuple[int, int, int]]:
    stride, voff, raw = primitive_layout(prim)
    if stride <= 0 or voff is None:
        return []

    kind = lname(prim)
    output: List[Tuple[int, int, int]] = []

    if kind == "triangles":
        count = int(prim.attrib.get("count", "0"))
        raw = raw[:count * 3 * stride]
        for i in range(0, len(raw), 3*stride):
            if i + 3*stride <= len(raw):
                output.append((
                    raw[i+voff],
                    raw[i+stride+voff],
                    raw[i+2*stride+voff],
                ))
    elif kind == "polylist":
        vcount = direct(prim, "vcount")
        if vcount is None:
            return []
        cursor = 0
        for n in inums(vcount.text):
            poly = [
                raw[cursor+j*stride+voff]
                for j in range(n)
                if cursor+j*stride+voff < len(raw)
            ]
            cursor += n*stride
            for j in range(1, len(poly)-1):
                output.append((poly[0], poly[j], poly[j+1]))
    return output


def connected_groups(npos: int, triangles: List[Tuple[int, int, int]]) -> List[set[int]]:
    valid = [t for t in triangles if all(0 <= i < npos for i in t)]
    used = sorted({i for t in valid for i in t})
    if not used:
        return []
    remap = {old: new for new, old in enumerate(used)}
    dsu = DSU(len(used))
    for a, b, c in valid:
        dsu.union(remap[a], remap[b])
        dsu.union(remap[b], remap[c])
    groups: Dict[int, set[int]] = defaultdict(set)
    for old, new in remap.items():
        groups[dsu.find(new)].add(old)
    return list(groups.values())


def find_left_frame_geometry(root: ET.Element) -> Tuple[str, set[int]]:
    """
    Find the geometry and position-index connected component whose transformed
    bounds match the old left C01 frame.
    """
    # Map geometry id -> (positions, triangles)
    geometry_data: Dict[str, Tuple[np.ndarray, List[Tuple[int, int, int]]]] = {}
    for geo in root.iter():
        if lname(geo) != "geometry":
            continue
        mesh = direct(geo, "mesh")
        if mesh is None:
            continue
        sources, pos_id = read_mesh_positions(mesh)
        if not pos_id or pos_id not in sources:
            continue
        tris: List[Tuple[int, int, int]] = []
        for p in list(mesh):
            if lname(p) in ("triangles", "polylist"):
                tris.extend(extract_triangles(p))
        if tris:
            geometry_data[geo.attrib.get("id", "")] = (sources[pos_id], tris)

    visual_scene = next((e for e in root.iter() if lname(e) == "visual_scene"), None)
    if visual_scene is None:
        raise RuntimeError("No visual_scene found in DAE.")

    best = None
    best_score = float("inf")

    def walk(node: ET.Element, parent: np.ndarray) -> None:
        nonlocal best, best_score
        m = parent @ node_transform(node)

        for child in list(node):
            if lname(child) == "instance_geometry":
                gid = child.attrib.get("url", "").lstrip("#")
                if gid not in geometry_data:
                    continue
                positions, tris = geometry_data[gid]
                for group in connected_groups(len(positions), tris):
                    idx = np.asarray(sorted(group), dtype=int)
                    pts = transform_points(positions[idx], m)
                    mn, mx = pts.min(axis=0), pts.max(axis=0)
                    b = np.array([mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]])
                    score = float(np.linalg.norm(b - TARGET_BOUNDS))
                    if score < best_score:
                        best_score = score
                        best = (gid, group, b)
            elif lname(child) == "node":
                walk(child, m)

    for child in list(visual_scene):
        if lname(child) == "node":
            walk(child, np.eye(4))

    if best is None or best_score > 0.05:
        raise RuntimeError(
            "Unable to identify original left frame C01 safely. "
            f"Best bounds score={best_score:.6f}; no files changed."
        )

    gid, group, bounds = best
    print(
        "Identified old left frame:\n"
        f"  geometry id: {gid}\n"
        f"  bounds: X[{bounds[0]:.6f}, {bounds[1]:.6f}], "
        f"Y[{bounds[2]:.6f}, {bounds[3]:.6f}], "
        f"Z[{bounds[4]:.6f}, {bounds[5]:.6f}]\n"
    )
    return gid, group


def strip_component_from_triangles(prim: ET.Element, target_positions: set[int]) -> int:
    """Filter triangle primitive records using vertex position indices."""
    stride, voff, raw = primitive_layout(prim)
    if stride <= 0 or voff is None or lname(prim) != "triangles":
        return 0

    count = int(prim.attrib.get("count", "0"))
    raw = raw[:count * 3 * stride]
    keep: List[int] = []
    removed = 0

    for i in range(0, len(raw), 3*stride):
        chunk = raw[i:i+3*stride]
        if len(chunk) != 3*stride:
            continue
        pos_indices = {
            chunk[voff],
            chunk[stride+voff],
            chunk[2*stride+voff],
        }
        if pos_indices and pos_indices.issubset(target_positions):
            removed += 1
        else:
            keep.extend(chunk)

    p_elems = [c for c in list(prim) if lname(c) == "p"]
    if len(p_elems) != 1:
        # This DAE has one p per triangle block in this project. Refuse if not.
        raise RuntimeError(
            "Unexpected DAE triangles layout: multiple <p> elements. "
            "No patch applied."
        )
    p_elems[0].text = fmt_ints(keep)
    prim.attrib["count"] = str(len(keep) // (3*stride))
    return removed


def patch_dae(source: Path, target: Path) -> int:
    tree = ET.parse(source)
    root = tree.getroot()
    geometry_id, target_positions = find_left_frame_geometry(root)

    removed_total = 0
    found = False
    for geo in root.iter():
        if lname(geo) != "geometry" or geo.attrib.get("id", "") != geometry_id:
            continue
        found = True
        mesh = direct(geo, "mesh")
        if mesh is None:
            continue
        for prim in list(mesh):
            if lname(prim) == "triangles":
                removed_total += strip_component_from_triangles(prim, target_positions)
            elif lname(prim) == "polylist":
                raise RuntimeError(
                    "The target geometry has polylist primitives. "
                    "This patcher intentionally refuses instead of risking a bad model."
                )

    if not found or removed_total <= 0:
        raise RuntimeError(
            "No left-frame triangles were removed. Aborting without output."
        )

    tree.write(
        target,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
    )
    return removed_total


def box_link(name: str, pose: str, size: str, color: str) -> str:
    return f"""
      <link name="{name}">
        <pose>{pose}</pose>
        <collision name="collision">
          <geometry><box><size>{size}</size></box></geometry>
          <surface>
            <contact><ode><min_depth>0.001</min_depth><max_vel>0</max_vel></ode></contact>
            <friction><ode/></friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{size}</size></box></geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
          </material>
        </visual>
      </link>"""


def write_model_sdf(path: Path) -> None:
    y_center = (GATE_MIN_Y + GATE_MAX_Y) / 2
    thickness = GATE_MAX_Y - GATE_MIN_Y
    # Side posts preserve an outer frame width, with the requested 0.98m opening.
    left_w = OPEN_MIN_X - GATE_MIN_X
    right_w = GATE_MAX_X - OPEN_MAX_X
    lower_h = OPEN_MIN_Z - GATE_MIN_Z
    top_h = GATE_MAX_Z - OPEN_MAX_Z
    color = "0.12 0.14 0.18 1"

    left_pose = f"{(GATE_MIN_X+OPEN_MIN_X)/2:.6f} {y_center:.6f} {(GATE_MIN_Z+GATE_MAX_Z)/2:.6f} 0 0 0"
    right_pose = f"{(OPEN_MAX_X+GATE_MAX_X)/2:.6f} {y_center:.6f} {(GATE_MIN_Z+GATE_MAX_Z)/2:.6f} 0 0 0"
    bottom_pose = f"{(OPEN_MIN_X+OPEN_MAX_X)/2:.6f} {y_center:.6f} {(GATE_MIN_Z+OPEN_MIN_Z)/2:.6f} 0 0 0"
    top_pose = f"{(OPEN_MIN_X+OPEN_MAX_X)/2:.6f} {y_center:.6f} {(OPEN_MAX_Z+GATE_MAX_Z)/2:.6f} 0 0 0"

    xml = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="test_world_gate_wide">
    <static>true</static>
    <link name="map">
      <gravity>true</gravity>
      <self_collide>false</self_collide>
      <pose>0 0 0 0 0 0</pose>
      <visual name="terrain_visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh><uri>model://test_world_gate_wide/meshes/test_terrain_wide_gate.dae</uri></mesh>
        </geometry>
      </visual>
      <collision name="terrain_collision">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh><uri>model://test_world_gate_wide/meshes/test_terrain_wide_gate.dae</uri></mesh>
        </geometry>
      </collision>
    </link>
{box_link("wide_gate_left_post", left_pose, f"{left_w:.6f} {thickness:.6f} {GATE_MAX_Z-GATE_MIN_Z:.6f}", color)}
{box_link("wide_gate_right_post", right_pose, f"{right_w:.6f} {thickness:.6f} {GATE_MAX_Z-GATE_MIN_Z:.6f}", color)}
{box_link("wide_gate_bottom_beam", bottom_pose, f"{OPEN_MAX_X-OPEN_MIN_X:.6f} {thickness:.6f} {lower_h:.6f}", color)}
{box_link("wide_gate_top_beam", top_pose, f"{OPEN_MAX_X-OPEN_MIN_X:.6f} {thickness:.6f} {top_h:.6f}", color)}
  </model>
</sdf>
"""
    path.write_text(xml, encoding="utf-8")


def write_world_sdf(path: Path) -> None:
    xml = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="test_world_gate_wide">
    <include>
      <name>test_world_gate_wide</name>
      <uri>model://test_world_gate_wide</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
"""
    path.write_text(xml, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--px4-root",
        default=str(Path.home() / "PX4-Autopilot"),
        help="PX4-Autopilot root",
    )
    args = ap.parse_args()

    px4 = Path(args.px4_root).expanduser().resolve()
    src_model = px4 / "Tools/simulation/gz/models/test_world"
    src_dae = src_model / "meshes/test_terrain.dae"
    models_dir = px4 / "Tools/simulation/gz/models"
    worlds_dir = px4 / "Tools/simulation/gz/worlds"

    dst_model = models_dir / "test_world_gate_wide"
    dst_mesh_dir = dst_model / "meshes"
    dst_dae = dst_mesh_dir / "test_terrain_wide_gate.dae"
    dst_world = worlds_dir / "test_world_gate_wide.sdf"

    if not src_dae.exists():
        print(f"ERROR: source terrain not found: {src_dae}", file=sys.stderr)
        return 2

    # Never overwrite an existing output model automatically.
    if dst_model.exists() or dst_world.exists():
        print("ERROR: target wide-gate world already exists:")
        print(f"  {dst_model}")
        print(f"  {dst_world}")
        print("Remove those two target paths manually only if you want to rebuild.")
        return 3

    print("Creating a NEW model and world. Original test_world is untouched.")
    print("Copying original test_world assets...")
    shutil.copytree(src_model, dst_model)

    # We only need the wide-patched terrain in the new model, but leave copied
    # original resources intact for textures / metadata compatibility.
    dst_mesh_dir.mkdir(exist_ok=True)

    print("Removing only the old narrow left-frame triangle component...")
    removed = patch_dae(src_dae, dst_dae)
    print(f"Removed {removed} old left-frame triangles from the new terrain mesh.")

    print("Writing explicit wide gate visual + collision geometry...")
    write_model_sdf(dst_model / "model.sdf")
    write_world_sdf(dst_world)

    print("\nSUCCESS")
    print("Original files unchanged:")
    print(f"  {src_model}")
    print("New wide-gate model:")
    print(f"  {dst_model}")
    print("New wide-gate world:")
    print(f"  {dst_world}")
    print("\nNew clear opening:")
    print(f"  width  = {OPEN_MAX_X-OPEN_MIN_X:.3f} m")
    print(f"  height = {OPEN_MAX_Z-OPEN_MIN_Z:.3f} m")
    print("  X range = %.3f .. %.3f m" % (OPEN_MIN_X, OPEN_MAX_X))
    print("  Z range = %.3f .. %.3f m" % (OPEN_MIN_Z, OPEN_MAX_Z))
    print("\nStart later using:")
    print('PX4_SYS_AUTOSTART=4010 \\')
    print('PX4_SIM_MODEL=gz_x500_mono_cam \\')
    print('PX4_GZ_MODEL_POSE="1,1,0.1,0,0,0" \\')
    print('PX4_GZ_WORLD=test_world_gate_wide \\')
    print('~/PX4-Autopilot/build/px4_sitl_default/bin/px4')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
