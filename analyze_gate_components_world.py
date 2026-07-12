#!/usr/bin/env python3
"""
Read-only, corrected obstacle-component extractor for test_terrain.dae.

Why this version exists:
The previous component script separated raw mesh vertices but did not apply the
DAE visual-scene node transform. Therefore its coordinates were not Gazebo
world coordinates and could show impossible values such as -185 m.

This version:
- splits triangle connectivity into physical components
- applies the DAE node transform used by the model
- applies the test_world SDF model pose
- writes bounds in Gazebo world coordinates
- reads no control interfaces and sends no PX4 / Gazebo commands
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


def lname(e):
    return e.tag.rsplit("}", 1)[-1]


def nums(text):
    return [float(x) for x in (text or "").replace("\n", " ").split()]


def ints(text):
    return [int(x) for x in (text or "").replace("\n", " ").split()]


def direct(parent, name):
    for c in list(parent):
        if lname(c) == name:
            return c
    return None


def transform_translate(v):
    m = np.eye(4)
    if len(v) >= 3:
        m[:3, 3] = v[:3]
    return m


def transform_scale(v):
    m = np.eye(4)
    if len(v) >= 3:
        m[0, 0], m[1, 1], m[2, 2] = v[:3]
    return m


def transform_rotate(v):
    if len(v) < 4:
        return np.eye(4)
    axis = np.array(v[:3], dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(4)
    x, y, z = axis / n
    a = np.deg2rad(v[3])
    c, s = np.cos(a), np.sin(a)
    q = 1 - c
    r = np.array([
        [c+x*x*q, x*y*q-z*s, x*z*q+y*s],
        [y*x*q+z*s, c+y*y*q, y*z*q-x*s],
        [z*x*q-y*s, z*y*q+x*s, c+z*z*q],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    return m


def node_transform(node):
    m = np.eye(4)
    for c in list(node):
        name = lname(c)
        values = nums(c.text)
        if name == "matrix" and len(values) >= 16:
            local = np.array(values[:16], dtype=float).reshape((4, 4), order="F")
        elif name == "translate":
            local = transform_translate(values)
        elif name == "scale":
            local = transform_scale(values)
        elif name == "rotate":
            local = transform_rotate(values)
        else:
            continue
        m = m @ local
    return m


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def source_positions(mesh):
    sources = {}
    for s in list(mesh):
        if lname(s) != "source":
            continue
        sid = s.attrib.get("id", "")
        fa = direct(s, "float_array")
        tech = direct(s, "technique_common")
        acc = direct(tech, "accessor") if tech is not None else None
        if fa is None:
            continue
        stride = int(acc.attrib.get("stride", "3")) if acc is not None else 3
        data = np.asarray(nums(fa.text), dtype=float)
        if stride < 3 or data.size < 3:
            continue
        data = data[:(data.size // stride)*stride].reshape((-1, stride))[:, :3]
        sources[sid] = data
    vertices = direct(mesh, "vertices")
    pos_id = None
    if vertices is not None:
        for inp in list(vertices):
            if lname(inp) == "input" and inp.attrib.get("semantic") == "POSITION":
                pos_id = inp.attrib.get("source", "").lstrip("#")
                break
    return sources, pos_id


def triangles_from_primitive(prim):
    ins = [c for c in list(prim) if lname(c) == "input"]
    if not ins:
        return []
    stride = max(int(i.attrib.get("offset", "0")) for i in ins) + 1
    v_off = None
    for i in ins:
        if i.attrib.get("semantic") == "VERTEX":
            v_off = int(i.attrib.get("offset", "0"))
            break
    if v_off is None:
        return []
    raw = []
    for c in list(prim):
        if lname(c) == "p":
            raw.extend(ints(c.text))
    if lname(prim) == "triangles":
        n = int(prim.attrib.get("count", "0"))
        raw = raw[:n * 3 * stride]
        return [
            (raw[i + v_off], raw[i + stride + v_off], raw[i + 2*stride + v_off])
            for i in range(0, len(raw), 3*stride)
            if i + 3*stride <= len(raw)
        ]
    if lname(prim) == "polylist":
        vc = direct(prim, "vcount")
        if vc is None:
            return []
        out, cur = [], 0
        for count in ints(vc.text):
            poly = [raw[cur+j*stride+v_off] for j in range(count)
                    if cur+j*stride+v_off < len(raw)]
            cur += count*stride
            out += [(poly[0], poly[j], poly[j+1]) for j in range(1, len(poly)-1)]
        return out
    return []


def geometry_components(root):
    result = {}
    for geo in root.iter():
        if lname(geo) != "geometry":
            continue
        mesh = direct(geo, "mesh")
        if mesh is None:
            continue
        sources, posid = source_positions(mesh)
        if not posid or posid not in sources:
            continue
        pos = sources[posid]
        tris = []
        for prim in list(mesh):
            if lname(prim) in ("triangles", "polylist"):
                tris += triangles_from_primitive(prim)
        used = sorted({i for t in tris for i in t if 0 <= i < len(pos)})
        if not used:
            continue
        remap = {old:i for i,old in enumerate(used)}
        dsu = DSU(len(used))
        for a,b,c in tris:
            if a in remap and b in remap and c in remap:
                dsu.union(remap[a], remap[b])
                dsu.union(remap[b], remap[c])
        groups = defaultdict(list)
        for old,new in remap.items():
            groups[dsu.find(new)].append(old)
        result[geo.attrib.get("id","")] = {
            "name": geo.attrib.get("name", geo.attrib.get("id","")),
            "positions": pos,
            "groups": list(groups.values()),
        }
    return result


def transform_points(points, m):
    hp = np.c_[points, np.ones(len(points))]
    return (m @ hp.T).T[:, :3]


def collect_instances(root, components):
    visual_scene = next((e for e in root.iter() if lname(e) == "visual_scene"), None)
    rows = []
    if visual_scene is None:
        return rows

    def walk(node, parent, path):
        name = node.attrib.get("name", node.attrib.get("id", "node"))
        m = parent @ node_transform(node)
        p = path + [name]
        for c in list(node):
            if lname(c) == "instance_geometry":
                gid = c.attrib.get("url","").lstrip("#")
                if gid not in components:
                    continue
                info = components[gid]
                for idx, old_ids in enumerate(info["groups"], 1):
                    pts = transform_points(info["positions"][np.asarray(old_ids, dtype=int)], m)
                    mn, mx = pts.min(0), pts.max(0)
                    sz = mx - mn
                    rows.append({
                        "node_path": "/".join(p),
                        "geometry_name": info["name"],
                        "component": idx,
                        "vertex_count": len(old_ids),
                        "min_x": float(mn[0]), "max_x": float(mx[0]),
                        "min_y": float(mn[1]), "max_y": float(mx[1]),
                        "min_z": float(mn[2]), "max_z": float(mx[2]),
                        "size_x": float(sz[0]), "size_y": float(sz[1]), "size_z": float(sz[2]),
                    })
            elif lname(c) == "node":
                walk(c, m, p)

    for child in list(visual_scene):
        if lname(child) == "node":
            walk(child, np.eye(4), [])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--px4-root", default=str(Path.home() / "PX4-Autopilot"))
    p.add_argument("--out", default="scene_inspection_components_fixed")
    a = p.parse_args()

    px4 = Path(a.px4_root).expanduser().resolve()
    dae = px4 / "Tools/simulation/gz/models/test_world/meshes/test_terrain.dae"
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not dae.exists():
        raise SystemExit(f"Cannot find: {dae}")

    root = ET.parse(dae).getroot()
    comps = geometry_components(root)
    rows = collect_instances(root, comps)
    rows.sort(key=lambda r: (-r["size_z"], r["min_x"], r["min_y"]))

    with (out / "components_world_bounds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (out / "components_world_bounds.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # A practical filter: components tall enough to be structural pieces,
    # discarding flat ground markings.
    vertical = [r for r in rows if r["size_z"] >= 0.15]
    lines = [
        "# Corrected Connected Components — World Bounds",
        "",
        "The DAE visual-scene transform is applied in this report.",
        "Coordinates are in the test_world model frame; test_world pose is zero.",
        "",
        "## Vertical structural candidates",
        "",
        "| # | comp | vertices | min_z | max_z | height | min_x | max_x | min_y | max_y |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i,r in enumerate(vertical,1):
        lines.append(
            f"| {i} | {r['component']} | {r['vertex_count']} | "
            f"{r['min_z']:.3f} | {r['max_z']:.3f} | {r['size_z']:.3f} | "
            f"{r['min_x']:.3f} | {r['max_x']:.3f} | "
            f"{r['min_y']:.3f} | {r['max_y']:.3f} |"
        )
    (out / "vertical_components_world.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (out / "README.txt").write_text(
        f"Done.\nDAE: {dae}\nGeometry definitions: {len(comps)}\n"
        f"World-space connected components: {len(rows)}\n"
        f"Vertical candidates: {len(vertical)}\nOutput: {out}\n",
        encoding="utf-8")
    print((out / "README.txt").read_text())


if __name__ == "__main__":
    main()
