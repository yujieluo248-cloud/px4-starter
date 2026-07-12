#!/usr/bin/env python3
"""
Repair the visual-material preservation for test_world_gate_wide.

Why this exists:
The first generator used xml.etree.ElementTree to rewrite the DAE. That parser
changed COLLADA namespace/material serialization. Geometry was retained, but
Gazebo no longer rendered the original terrain materials/textures correctly.

This repair:
  1) patches the existing generator to use lxml (which preserves COLLADA
     namespace prefixes and material bindings),
  2) backs up ONLY the generated wide-gate model/world,
  3) rebuilds the generated wide-gate model/world from the untouched original
     test_world source.

It never modifies the original:
  ~/PX4-Autopilot/Tools/simulation/gz/models/test_world
  ~/PX4-Autopilot/Tools/simulation/gz/worlds/test_world.sdf
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    project = Path.home() / "PX4-ROS2-Gazebo-Drone-Simulation-Template"
    generator = project / "create_wide_gate_world.py"
    px4 = Path.home() / "PX4-Autopilot"

    if not generator.exists():
        die(f"Generator not found: {generator}")

    # This verification is intentionally before touching any generated assets.
    try:
        import lxml.etree  # noqa: F401
    except ImportError:
        die(
            "python3-lxml is not installed. Run this first:\n"
            "  sudo apt-get update && sudo apt-get install -y python3-lxml\n"
            "Then run this repair script again."
        )

    source = generator.read_text(encoding="utf-8")
    old_import = "import xml.etree.ElementTree as ET"
    new_import = "from lxml import etree as ET"

    old_write = (
        '    ET.indent(tree, space="  ")\n'
        '    tree.write(target, encoding="utf-8", xml_declaration=True)'
    )
    new_write = (
        '    tree.write(\n'
        '        target,\n'
        '        encoding="utf-8",\n'
        '        xml_declaration=True,\n'
        '        pretty_print=True,\n'
        '    )'
    )

    if old_import in source:
        source = source.replace(old_import, new_import, 1)
        print("Patched generator: ElementTree -> lxml.")
    elif new_import in source:
        print("Generator already uses lxml.")
    else:
        die("Could not recognize the XML import in the generator; refusing to patch.")

    if old_write in source:
        source = source.replace(old_write, new_write, 1)
        print("Patched generator: lxml-safe DAE writer enabled.")
    elif "pretty_print=True" in source:
        print("Generator already has lxml-safe write settings.")
    else:
        die("Could not recognize the DAE write block; refusing to patch.")

    backup_generator = generator.with_suffix(".py.before_lxml_fix")
    if not backup_generator.exists():
        shutil.copy2(generator, backup_generator)
        print(f"Backed up original generator: {backup_generator.name}")

    generator.write_text(source, encoding="utf-8")

    # Confirm patched script syntax before changing generated scene files.
    check = subprocess.run(
        [sys.executable, "-m", "py_compile", str(generator)],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        die("Patched generator failed syntax check:\n" + check.stderr)

    generated_model = px4 / "Tools/simulation/gz/models/test_world_gate_wide"
    generated_world = px4 / "Tools/simulation/gz/worlds/test_world_gate_wide.sdf"

    if not generated_model.exists() or not generated_world.exists():
        die(
            "Expected generated wide-gate assets were not found. "
            "Nothing was removed."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = px4 / "Tools/simulation/gz/wide_gate_material_backup" / stamp
    backup_root.mkdir(parents=True, exist_ok=False)

    shutil.move(str(generated_model), str(backup_root / generated_model.name))
    shutil.move(str(generated_world), str(backup_root / generated_world.name))
    print(f"Backed up current generated scene to: {backup_root}")

    print("Rebuilding wide-gate scene from the untouched original model...")
    result = subprocess.run(
        [sys.executable, str(generator), "--px4-root", str(px4)],
        cwd=str(project),
        text=True,
    )
    if result.returncode != 0:
        die(
            "Rebuild failed. The previous generated scene is safely stored at:\n"
            f"  {backup_root}\n"
            "Original test_world remains untouched."
        )

    dae = generated_model / "meshes/test_terrain_wide_gate.dae"
    if not dae.exists():
        die("Rebuild reported success but patched DAE is missing.")

    first = dae.read_text(encoding="utf-8", errors="replace")[:1000]
    if "<COLLADA" not in first:
        die(
            "New DAE does not begin with the expected COLLADA root tag. "
            "Do not launch Gazebo; inspect the rebuild output."
        )

    print("\nSUCCESS")
    print("Rebuilt the wide-gate scene with original COLLADA material bindings preserved.")
    print("Start Gazebo again with PX4_GZ_WORLD=test_world_gate_wide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
