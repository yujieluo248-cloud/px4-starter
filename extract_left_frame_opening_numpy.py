#!/usr/bin/env python3
"""
Dependency-light, read-only left-frame opening extractor.

This version intentionally does NOT use OpenCV, matplotlib, SciPy, or any
compiled extension beyond NumPy. It is designed for environments where
python3-opencv was compiled against NumPy 1.x while Python uses NumPy 2.x.

It parses the DAE mesh, applies the DAE visual-scene transforms, isolates the
known left frame, projects its triangles into the X-Z plane, rasterizes the
projection using a pure-Python scanline method, and finds enclosed free-space
regions.

It never controls PX4, ROS 2, Gazebo, or the UAV.
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


def lname(e):
    return e.tag.rsplit("}", 1)[-1]


def direct(parent, name):
    if parent is None:
        return None
    for c in list(parent):
        if lname(c) == name:
            return c
    return None


def fnums(text):
    return [float(x) for x in (text or "").replace("\n", " ").split()]


def inums(text):
    return [int(x) for x in (text or "").replace("\n", " ").split()]


def translation(v):
    m = np.eye(4)
    if len(v) >= 3:
        m[:3, 3] = v[:3]
    return m


def scaling(v):
    m = np.eye(4)
    if len(v) >= 3:
        m[0, 0], m[1, 1], m[2, 2] = v[:3]
    return m


def axis_angle(v):
    if len(v) < 4:
        return np.eye(4)
    axis = np.asarray(v[:3], dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(4)
    x, y, z = axis / n
    a = math.radians(v[3])
    c, s, q = math.cos(a), math.sin(a), 1 - math.cos(a)
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
        n, v = lname(c), fnums(c.text)
        if n == "matrix" and len(v) >= 16:
            local = np.asarray(v[:16], dtype=float).reshape((4, 4), order="F")
        elif n == "translate":
            local = translation(v)
        elif n == "scale":
            local = scaling(v)
        elif n == "rotate":
            local = axis_angle(v)
        else:
            continue
        m = m @ local
    return m


def transform_points(points, m):
    h = np.c_[points, np.ones(len(points))]
    return (m @ h.T).T[:, :3]


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


def read_sources(mesh):
    sources = {}
    for src in list(mesh):
        if lname(src) != "source":
            continue
        sid = src.attrib.get("id", "")
        fa = direct(src, "float_array")
        tech = direct(src, "technique_common")
        acc = direct(tech, "accessor")
        if fa is None:
            continue
        stride = int(acc.attrib.get("stride", "3")) if acc is not None else 3
        a = np.asarray(fnums(fa.text), dtype=float)
        if stride < 3 or a.size < stride:
            continue
        sources[sid] = a[:(a.size // stride)*stride].reshape((-1, stride))[:, :3]

    vertices = direct(mesh, "vertices")
    posid = None
    if vertices is not None:
        for inp in list(vertices):
            if lname(inp) == "input" and inp.attrib.get("semantic") == "POSITION":
                posid = inp.attrib.get("source", "").lstrip("#")
                break
    return sources, posid


def triangles_from_prim(prim):
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
            raw.extend(inums(c.text))

    if lname(prim) == "triangles":
        cnt = int(prim.attrib.get("count", "0"))
        raw = raw[:cnt * 3 * stride]
        return [
            (raw[i+v_off], raw[i+stride+v_off], raw[i+2*stride+v_off])
            for i in range(0, len(raw), 3*stride)
            if i + 3*stride <= len(raw)
        ]

    if lname(prim) == "polylist":
        vc = direct(prim, "vcount")
        if vc is None:
            return []
        result, cursor = [], 0
        for n in inums(vc.text):
            poly = [
                raw[cursor+j*stride+v_off]
                for j in range(n)
                if cursor+j*stride+v_off < len(raw)
            ]
            cursor += n * stride
            for j in range(1, len(poly)-1):
                result.append((poly[0], poly[j], poly[j+1]))
        return result
    return []


def load_geometry(root):
    output = {}
    for geo in root.iter():
        if lname(geo) != "geometry":
            continue
        mesh = direct(geo, "mesh")
        if mesh is None:
            continue
        sources, posid = read_sources(mesh)
        if not posid or posid not in sources:
            continue
        tris = []
        for p in list(mesh):
            if lname(p) in ("triangles", "polylist"):
                tris.extend(triangles_from_prim(p))
        if tris:
            output[geo.attrib.get("id", "")] = {
                "positions": sources[posid],
                "triangles": tris,
                "name": geo.attrib.get("name", geo.attrib.get("id", "")),
            }
    return output


def connected_groups(npos, tris):
    valid = [t for t in tris if all(0 <= i < npos for i in t)]
    used = sorted({i for t in valid for i in t})
    if not used:
        return []
    remap = {old:new for new, old in enumerate(used)}
    dsu = DSU(len(used))
    for a, b, c in valid:
        dsu.union(remap[a], remap[b])
        dsu.union(remap[b], remap[c])
    groups = defaultdict(set)
    for old, new in remap.items():
        groups[dsu.find(new)].add(old)
    return list(groups.values())


def instances_from_scene(root, geometries):
    scene = next((e for e in root.iter() if lname(e) == "visual_scene"), None)
    if scene is None:
        return []
    out = []

    def walk(node, parent_m, path):
        nm = node.attrib.get("name", node.attrib.get("id", "node"))
        m = parent_m @ node_transform(node)
        current = path + [nm]
        for child in list(node):
            if lname(child) == "instance_geometry":
                gid = child.attrib.get("url", "").lstrip("#")
                if gid not in geometries:
                    continue
                geo = geometries[gid]
                for comp_id, group in enumerate(connected_groups(len(geo["positions"]), geo["triangles"]), start=1):
                    ids = np.asarray(sorted(group), dtype=int)
                    pts = transform_points(geo["positions"][ids], m)
                    tri = [t for t in geo["triangles"] if t[0] in group and t[1] in group and t[2] in group]
                    out.append({
                        "component_id": comp_id,
                        "path": "/".join(current),
                        "positions": geo["positions"],
                        "matrix": m,
                        "triangles": tri,
                        "min": pts.min(axis=0),
                        "max": pts.max(axis=0),
                    })
            elif lname(child) == "node":
                walk(child, m, current)

    for child in list(scene):
        if lname(child) == "node":
            walk(child, np.eye(4), [])
    return out


def choose_left_frame(instances):
    target = np.array([-0.060, 1.140, 3.078, 3.378, 0.000, 1.853])
    best = None
    best_score = float("inf")
    for it in instances:
        mn, mx = it["min"], it["max"]
        v = np.array([mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]])
        score = float(np.linalg.norm(v - target))
        if score < best_score:
            best_score, best = score, it
    if best is None:
        raise RuntimeError("Unable to locate left frame.")
    return best


def world_triangles(inst):
    p, m = inst["positions"], inst["matrix"]
    return np.asarray([
        transform_points(p[np.asarray([a, b, c], dtype=int)], m)
        for a, b, c in inst["triangles"]
    ], dtype=float)


def fill_triangle(mask, tri_uv):
    """
    Pure Python scanline triangle fill. tri_uv is 3x2 float [u, v].
    """
    h, w = mask.shape
    xs = tri_uv[:, 0]
    ys = tri_uv[:, 1]
    min_y = max(0, int(math.floor(float(np.min(ys)))))
    max_y = min(h - 1, int(math.ceil(float(np.max(ys)))))

    if max_y < min_y:
        return

    verts = [(float(xs[i]), float(ys[i])) for i in range(3)]

    for row in range(min_y, max_y + 1):
        scan_y = row + 0.5
        crossings = []
        for i in range(3):
            x1, y1 = verts[i]
            x2, y2 = verts[(i+1) % 3]
            # Half-open edge rule avoids double-count at vertices.
            if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                if y2 != y1:
                    x = x1 + (scan_y-y1)*(x2-x1)/(y2-y1)
                    crossings.append(x)
        crossings.sort()
        for i in range(0, len(crossings)-1, 2):
            left = max(0, int(math.ceil(crossings[i] - 0.5)))
            right = min(w - 1, int(math.floor(crossings[i+1] - 0.5)))
            if right >= left:
                mask[row, left:right+1] = True


def rasterize(tris, mn, mx, pixels_long):
    width = float(mx[0] - mn[0])
    height = float(mx[2] - mn[2])
    ppm = pixels_long / max(width, height)
    w = max(120, int(math.ceil(width * ppm)) + 6)
    h = max(120, int(math.ceil(height * ppm)) + 6)
    min_x = float(mn[0] - 3/ppm)
    max_z = float(mx[2] + 3/ppm)

    material = np.zeros((h, w), dtype=bool)
    for tri in tris:
        # x grows right; image row grows downward, so z is flipped.
        uv = np.c_[
            (tri[:, 0] - min_x) * ppm,
            (max_z - tri[:, 2]) * ppm,
        ]
        fill_triangle(material, uv)
    return material, ppm, min_x, max_z


def external_free(material):
    h, w = material.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()

    def push(r, c):
        if 0 <= r < h and 0 <= c < w and not material[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))

    for c in range(w):
        push(0, c); push(h-1, c)
    for r in range(h):
        push(r, 0); push(r, w-1)

    while q:
        r, c = q.popleft()
        push(r-1, c); push(r+1, c); push(r, c-1); push(r, c+1)
    return seen


def connected_regions(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    regions = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr,cc] and not seen[rr,cc]:
                        seen[rr,cc] = True
                        q.append((rr,cc))
            regions.append(cells)
    return regions


def make_pgm(path, material, openings):
    """
    Portable GrayMap. Open it with any image viewer or with:
      xdg-open <file>.pgm
    black=material, gray=outside air, white=opening
    """
    img = np.full(material.shape, 175, dtype=np.uint8)
    img[material] = 20
    img[openings] = 245
    with path.open("wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px4-root", default=str(Path.home() / "PX4-Autopilot"))
    ap.add_argument("--out", default="scene_inspection_openings_numpy")
    ap.add_argument("--pixels-long", type=int, default=800)
    args = ap.parse_args()

    px4 = Path(args.px4_root).expanduser().resolve()
    dae = px4 / "Tools/simulation/gz/models/test_world/meshes/test_terrain.dae"
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not dae.exists():
        print(f"ERROR: missing DAE: {dae}", file=sys.stderr)
        return 2

    root = ET.parse(dae).getroot()
    inst = choose_left_frame(instances_from_scene(root, load_geometry(root)))
    tris = world_triangles(inst)
    mn, mx = inst["min"], inst["max"]

    material, ppm, min_x, max_z = rasterize(tris, mn, mx, max(300, args.pixels_long))
    exterior = external_free(material)
    opening_mask = (~material) & (~exterior)
    regions = connected_regions(opening_mask)

    rows = []
    for cells in regions:
        if len(cells) < 20:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        low_x = min_x + min(cs)/ppm
        high_x = min_x + (max(cs)+1)/ppm
        high_z = max_z - min(rs)/ppm
        low_z = max_z - (max(rs)+1)/ppm
        rows.append({
            "opening_id": len(rows)+1,
            "cell_count": len(cells),
            "min_x": float(low_x),
            "max_x": float(high_x),
            "min_z": float(low_z),
            "max_z": float(high_z),
            "width_x": float(high_x-low_x),
            "height_z": float(high_z-low_z),
            "center_x": float((low_x+high_x)/2),
            "center_y": float((mn[1]+mx[1])/2),
            "center_z": float((low_z+high_z)/2),
            "frame_min_y": float(mn[1]),
            "frame_max_y": float(mx[1]),
            "pixels_per_meter": float(ppm),
        })
    rows.sort(key=lambda r: r["width_x"]*r["height_z"], reverse=True)

    with (out/"left_frame_openings.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else ["opening_id","cell_count","min_x","max_x","min_z","max_z","width_x","height_z","center_x","center_y","center_z","frame_min_y","frame_max_y","pixels_per_meter"]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    (out/"left_frame_openings.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    make_pgm(out/"left_frame_opening_map.pgm", material, opening_mask)

    lines = [
        "# Left Frame Opening Extraction — NumPy / Pure Python",
        "",
        f"Source DAE: `{dae}`",
        "",
        "## Selected left-frame outer bounds",
        f"- X: {mn[0]:.6f} to {mx[0]:.6f} m",
        f"- Y: {mn[1]:.6f} to {mx[1]:.6f} m",
        f"- Z: {mn[2]:.6f} to {mx[2]:.6f} m",
        f"- Raster scale: {ppm:.2f} pixels/m",
        "",
        "## Detected enclosed openings",
        "",
        "| rank | cells | width X (m) | height Z (m) | center X | center Y | center Z | min Z | max Z |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {r['cell_count']} | {r['width_x']:.4f} | {r['height_z']:.4f} | "
            f"{r['center_x']:.4f} | {r['center_y']:.4f} | {r['center_z']:.4f} | "
            f"{r['min_z']:.4f} | {r['max_z']:.4f} |"
        )
    if not rows:
        lines.append("| - | - | No enclosed opening detected | | | | | | |")
    lines += [
        "",
        "## Map file",
        "- `left_frame_opening_map.pgm` — black: material; gray: exterior air; white: opening.",
        "- This image format needs no Python imaging library. Open with `xdg-open` or any image viewer.",
        "",
        "This script is read-only. No flight-control or Gazebo service command was sent.",
    ]
    (out/"left_frame_openings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (out/"README.txt").write_text(
        f"Done.\nDAE: {dae}\n"
        f"Outer bounds: X[{mn[0]:.3f}, {mx[0]:.3f}], Y[{mn[1]:.3f}, {mx[1]:.3f}], Z[{mn[2]:.3f}, {mx[2]:.3f}]\n"
        f"Triangles rasterized: {len(tris)}\n"
        f"Detected enclosed openings: {len(rows)}\n"
        f"Output: {out}\n", encoding="utf-8"
    )
    print((out/"README.txt").read_text())


if __name__ == "__main__":
    raise SystemExit(main())
