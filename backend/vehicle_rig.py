"""Vehicle rigging: partition a fused generated mesh into chassis + wheel nodes.

Face-partition approach — no boolean ops: faces whose centroids fall inside a
wheel's cutting cylinder become that wheel's mesh; the remainder is the chassis.
Boundary loops on both sides are capped by centroid fan. Wheel vertices are
recentered so each wheel node's pivot sits on its axle.
"""
from __future__ import annotations

import math
from io import BytesIO

import numpy as np
import trimesh

from .models import VehicleRigSpec, VehicleWheelSpec


def load_single_mesh(data: bytes) -> trimesh.Trimesh:
    loaded = trimesh.load(BytesIO(data), file_type="glb", force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("artifact contains no usable triangle mesh")
    return trimesh.util.concatenate(meshes)


def strip_ground_plane(mesh: trimesh.Trimesh, passes: int = 2) -> tuple[trimesh.Trimesh, int]:
    """Remove the flat floor sheet generations often keep: faces fully inside a
    thin layer at min-Y with near-vertical normals. Runs multiple passes to catch
    slightly-thick or double-layer planes."""
    working = mesh
    removed = 0
    for _ in range(passes):
        verts = np.asarray(working.vertices)
        y_min = float(verts[:, 1].min())
        y_span = float(verts[:, 1].max() - y_min)
        if y_span <= 0:
            break
        eps = max(1e-4, 0.02 * y_span)
        face_ys = verts[working.faces][:, :, 1]
        drop = (face_ys < y_min + eps).all(axis=1) & (np.abs(working.face_normals[:, 1]) > 0.8)
        if not drop.any():
            break
        removed += int(drop.sum())
        working = _submesh_with_uv(working, ~drop)
    return working, removed


def _axis_vector(axis: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        raise ValueError("wheel axis must be non-zero")
    return vec / norm


def _inside_cylinder(centroids: np.ndarray, wheel: VehicleWheelSpec) -> np.ndarray:
    center = np.asarray(wheel.center, dtype=float)
    axis = _axis_vector(wheel.axis)
    rel = centroids - center
    along = rel @ axis
    radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    return (radial < wheel.radius) & (np.abs(along) < wheel.half_width)


def suggest_wheel_regions(mesh: trimesh.Trimesh) -> list[VehicleWheelSpec]:
    """Auto-guess four wheel cylinders from bottom-region geometry.

    Quadrant clusters of the lowest 30% of the mesh get a two-pass cylinder fit.
    Both candidate axle axes (X and Z) are tried per quadrant; the fit whose
    capture is thin on exactly one axis wins. Always a suggestion — callers must
    let the user correct the result.
    """
    verts = np.asarray(mesh.vertices)
    y_min = float(verts[:, 1].min())
    y_span = float(verts[:, 1].max() - y_min)
    if y_span <= 0:
        return []
    band = verts[verts[:, 1] < y_min + 0.30 * y_span]
    if len(band) < 8:
        return []
    mx, mz = np.median(band[:, 0]), np.median(band[:, 2])
    centroids = mesh.triangles_center
    face_normals = mesh.face_normals
    up = np.array([0.0, 1.0, 0.0])
    wheels: list[VehicleWheelSpec] = []
    for left in (True, False):
        for front in (True, False):
            sel = band[(band[:, 0] < mx) == left]
            sel = sel[(sel[:, 2] < mz) == (not front)]
            if len(sel) < 4:
                continue
            best = None
            for axis in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
                a = _axis_vector(axis)
                other = np.cross(a, up)  # horizontal disk axis
                low = sel[sel[:, 1] < np.percentile(sel[:, 1], 60)]
                cy = float(np.mean(low[:, 1]))
                co = float(np.mean(low @ other))
                near = sel
                radius = 0.0
                for _ in range(2):
                    rel_y = sel[:, 1] - cy
                    disk_off = (sel @ other) - co
                    radial = np.sqrt(disk_off**2 + rel_y**2)
                    radius = float(np.percentile(radial, 80))
                    along = sel @ a
                    near = sel[(radial < radius * 1.1) & (np.abs(along - float(np.median(along))) < radius)]
                    if len(near) < 4:
                        break
                    cy = float(np.mean(near[:, 1]))
                    co = float(np.mean(near @ other))
                    radial = np.sqrt(((near @ other) - co) ** 2 + (near[:, 1] - cy) ** 2)
                    radius = float(np.percentile(radial, 85))
                if len(near) < 4 or radius <= 0:
                    continue
                ca = float(np.mean(near @ a))
                a_span = float(np.ptp(near @ a))
                half_width = float(np.clip(a_span / 2.0, radius * 0.2, radius * 0.6))
                center = ca * a + cy * up + co * other
                wheel = VehicleWheelSpec(name="candidate", center=tuple(float(v) for v in center), axis=axis, radius=radius, half_width=half_width)
                inside = _inside_cylinder(centroids, wheel)
                if inside.sum() < 20:
                    continue
                part = trimesh.Trimesh(vertices=verts, faces=mesh.faces[inside], process=False)
                part.remove_unreferenced_vertices()
                extents = np.ptp(part.vertices, axis=0)
                axle_idx = int(np.argmax(np.abs(a)))
                thin = float(np.delete(extents, axle_idx).max()) - float(extents[axle_idx])
                # A wheel's curved surface points radially — normals mostly
                # perpendicular to the true axle. A wrong-axis slab catches
                # faces that face along the axis; penalize those captures.
                axial_frac = float((np.abs(face_normals[inside] @ a) > 0.7).mean())
                score = thin - axial_frac
                if best is None or score > best[0]:
                    best = (score, wheel)
            if best is not None:
                wheels.append(best[1])
    names = {(True, True): "Wheel_FL", (True, False): "Wheel_RL", (False, True): "Wheel_FR", (False, False): "Wheel_RR"}
    for wheel in wheels:
        axle_idx = int(np.argmax(np.abs(np.asarray(wheel.axis))))
        # Axle axis is lateral; the other horizontal axis runs bumper-to-bumper.
        length_idx = 2 if axle_idx == 0 else 0
        lateral_med = mx if axle_idx == 0 else mz
        length_med = mz if length_idx == 2 else mx
        wheel.name = names[(wheel.center[axle_idx] < lateral_med, wheel.center[length_idx] > length_med)]
        wheel.steer = wheel.name.startswith("Wheel_F")
    return wheels


def _boundary_loops(part: trimesh.Trimesh) -> list[np.ndarray]:
    """Ordered vertex-index loops of open boundary edges."""
    ue = part.edges_unique
    counts = np.bincount(part.edges_unique_inverse, minlength=len(ue))
    boundary = ue[counts == 1]
    if len(boundary) == 0:
        return []
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    loops: list[np.ndarray] = []
    visited: set[tuple[int, int]] = set()
    for a, b in boundary:
        edge = (min(int(a), int(b)), max(int(a), int(b)))
        if edge in visited:
            continue
        visited.add(edge)
        start = int(a)
        loop = [start, int(b)]
        prev, cur = start, int(b)
        # Each step must consume a fresh boundary edge — bounds the walk and
        # prevents junction vertices from cycling the path forever.
        while True:
            neighbors = [n for n in adjacency.get(cur, []) if n != prev and (min(cur, n), max(cur, n)) not in visited]
            if not neighbors:
                break
            nxt = neighbors[0]
            visited.add((min(cur, nxt), max(cur, nxt)))
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.asarray(loop))
    return loops


def _cap_boundary(part: trimesh.Trimesh) -> int:
    """Fan-triangulate each open boundary loop. Returns faces added.

    New center vertices inherit the loop's mean UV so textured meshes keep a
    consistent vertex/UV pairing through export."""
    loops = _boundary_loops(part)
    if not loops:
        return 0
    visual = getattr(part, "visual", None)
    uv = getattr(visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(part.vertices)
    new_verts: list[np.ndarray] = []
    new_uvs: list[np.ndarray] = []
    new_faces: list[list[int]] = []
    base = len(part.vertices)
    for loop in loops:
        loop_verts = np.asarray(part.vertices)[loop]
        center = loop_verts.mean(axis=0)
        ci = base + len(new_verts)
        new_verts.append(center)
        if has_uv:
            new_uvs.append(np.asarray(uv)[loop].mean(axis=0))
        for i in range(len(loop)):
            new_faces.append([int(loop[i]), int(loop[(i + 1) % len(loop)]), ci])
    if not new_faces:
        return 0
    part.vertices = np.vstack([part.vertices, np.asarray(new_verts)])
    part.faces = np.vstack([part.faces, np.asarray(new_faces)])
    if has_uv:
        part.visual = trimesh.visual.TextureVisuals(uv=np.vstack([np.asarray(uv), np.asarray(new_uvs)]), material=getattr(visual, "material", None))
    return len(new_faces)


def _submesh_with_uv(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh:
    """Slice faces out preserving per-vertex UVs via exact index remap."""
    faces = mesh.faces[face_mask]
    unique, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    part = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[unique], faces=inverse.reshape(-1, 3), process=False)
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is not None and len(uv) == len(mesh.vertices):
        part.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv)[unique], material=getattr(visual, "material", None))
    return part


def partition_vehicle(mesh: trimesh.Trimesh, spec: VehicleRigSpec) -> tuple[trimesh.Scene, dict]:
    """Partition faces per wheel cylinder, cap boundaries, recenter pivots.

    Returns (scene, report). The scene contains `Chassis` plus one node per
    wheel; wheel node transforms carry the pivot so the geometry is centered on
    its axle. Report carries per-wheel face counts and cap counts for telemetry
    and honest-failure checks.
    """
    working = mesh
    removed = 0
    if spec.strip_ground:
        working, removed = strip_ground_plane(working)
    centroids = working.triangles_center
    hits = np.zeros((working.faces.shape[0], len(spec.wheels)), dtype=bool)
    for wi, wheel in enumerate(spec.wheels):
        hits[:, wi] = _inside_cylinder(centroids, wheel)
    overlap_faces = int((hits.sum(axis=1) > 1).sum())
    assignment = np.full(working.faces.shape[0], -1, dtype=int)
    for wi in range(len(spec.wheels)):
        assignment[hits[:, wi]] = wi
    scene = trimesh.Scene()
    report = {"ground_faces_removed": removed, "wheels": [], "chassis_faces": int((assignment == -1).sum())}
    if overlap_faces:
        report.setdefault("warnings", []).append(f"overlapping-wheel-cylinders-{overlap_faces}-faces")
    for wi, wheel in enumerate(spec.wheels):
        part = _submesh_with_uv(working, assignment == wi)
        capped = _cap_boundary(part)
        pivot = np.asarray(wheel.center, dtype=float)
        part.vertices = np.asarray(part.vertices) - pivot
        transform = np.eye(4)
        transform[:3, 3] = pivot
        scene.add_geometry(part, node_name=wheel.name, geom_name=wheel.name, transform=transform)
        report["wheels"].append({"name": wheel.name, "faces": int(part.faces.shape[0]), "cap_faces": capped, "pivot": [float(v) for v in pivot]})
        if part.faces.shape[0] < 20:
            report.setdefault("warnings", []).append(f"wheel-{wheel.name}-few-faces")
    chassis = _submesh_with_uv(working, assignment == -1)
    chassis_caps = _cap_boundary(chassis)
    scene.add_geometry(chassis, node_name="Chassis", geom_name="Chassis", transform=np.eye(4))
    report["chassis_cap_faces"] = chassis_caps
    if not spec.wheels or all(entry["faces"] == 0 for entry in report["wheels"]):
        raise ValueError("No wheel region contained geometry — reposition markers")
    if any(entry["faces"] == 0 for entry in report["wheels"]):
        empty = [entry["name"] for entry in report["wheels"] if entry["faces"] == 0]
        raise ValueError(f"Wheel regions captured no geometry: {', '.join(empty)}")
    return scene, report


def build_rigged_glb(data: bytes, spec: VehicleRigSpec) -> tuple[bytes, dict]:
    """Full rig build: load -> strip -> partition -> cap -> scene GLB."""
    mesh = load_single_mesh(data)
    scene, report = partition_vehicle(mesh, spec)
    return bytes(scene.export(file_type="glb")), report


def rigged_node_summary(data: bytes) -> dict:
    """Reload a rigged GLB and report node names + pivots (for validation)."""
    scene = trimesh.load(BytesIO(data), file_type="glb", force="scene")
    nodes = []
    for node in scene.graph.nodes_geometry:
        transform, geom = scene.graph.get(node)
        nodes.append({"node": node, "geometry": geom, "pivot": [float(v) for v in transform[:3, 3]]})
    return {"nodes": nodes}


def rig_quality_checks(data: bytes, spec: VehicleRigSpec) -> tuple[list[str], list[str]]:
    """Validation checks for a rigged artifact: (checks, blocking_failures)."""
    checks: list[str] = []
    blocking: list[str] = []
    try:
        summary = rigged_node_summary(data)
    except Exception:
        return checks, ["rig-glb-unreadable"]
    names = {entry["node"] for entry in summary["nodes"]}
    checks.append("rig-glb-readable")
    if "Chassis" in names:
        checks.append("rig-chassis-present")
    else:
        blocking.append("rig-chassis-missing")
    pivots = {entry["node"]: entry["pivot"] for entry in summary["nodes"]}
    for wheel in spec.wheels:
        if wheel.name not in names:
            blocking.append(f"rig-wheel-missing-{wheel.name}")
            continue
        checks.append(f"rig-wheel-{wheel.name}")
        pivot = np.asarray(pivots[wheel.name])
        expected = np.asarray(wheel.center)
        if not np.allclose(pivot, expected, atol=1e-3):
            blocking.append(f"rig-pivot-mismatch-{wheel.name}")
        else:
            checks.append(f"rig-pivot-{wheel.name}")
    for i, first in enumerate(spec.wheels):
        for second in spec.wheels[i + 1:]:
            gap = float(np.linalg.norm(np.asarray(first.center) - np.asarray(second.center)))
            if gap < (first.radius + second.radius) * 0.8:
                blocking.append(f"rig-wheels-overlap-{first.name}-{second.name}")
    return checks, blocking
