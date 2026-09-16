"""Phase 0 spike: prove face-partition vehicle rigging on a real generated GLB.

Usage: python scripts/spike_vehicle_rig.py <input.glb> [output.glb]

Steps: load -> suggest wheel cylinders -> partition faces -> cap -> recenter
pivots -> export named-node scene -> reload and verify hierarchy.
"""
import sys
from io import BytesIO

import numpy as np
import trimesh


def load_single_mesh(data: bytes) -> trimesh.Trimesh:
    loaded = trimesh.load(BytesIO(data), file_type="glb", force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("no meshes")
    return trimesh.util.concatenate(meshes)


def strip_ground_plane(mesh: trimesh.Trimesh, eps_frac: float = 0.02):
    """Remove the flat floor sheet generations often keep: faces fully inside a
    thin layer at min-Y whose normals are near-vertical. Returns cleaned mesh."""
    verts = np.asarray(mesh.vertices)
    y_min = float(verts[:, 1].min())
    y_span = float(verts[:, 1].max() - y_min)
    eps = max(1e-4, eps_frac * y_span)
    face_ys = verts[mesh.faces][:, :, 1]
    low = (face_ys < y_min + eps).all(axis=1)
    normals = mesh.face_normals
    flat = np.abs(normals[:, 1]) > 0.8
    drop = low & flat
    kept = trimesh.Trimesh(vertices=verts.copy(), faces=mesh.faces[~drop], process=False)
    kept.remove_unreferenced_vertices()
    # a second pass catches a slightly-thick or double-layer plane
    verts2 = np.asarray(kept.vertices)
    y_min2 = float(verts2[:, 1].min())
    face_ys2 = verts2[kept.faces][:, :, 1]
    drop2 = (face_ys2 < y_min2 + eps).all(axis=1) & (np.abs(kept.face_normals[:, 1]) > 0.8)
    kept2 = trimesh.Trimesh(vertices=verts2.copy(), faces=kept.faces[~drop2], process=False)
    kept2.remove_unreferenced_vertices()
    return kept2, int(drop.sum() + drop2.sum())


def suggest_wheels(mesh: trimesh.Trimesh, n_wheels: int = 4):
    """Cluster bottom-region vertices into XZ quadrants; fit a cylinder per cluster.

    Returns list of dicts: {name, center[x,y,z], axis, radius, half_width}.
    Axis assumed lateral (X) for a vehicle facing +-Z.
    Two-pass: coarse quadrant guess, then refine on verts near the estimate so
    fender/floor geometry doesn't inflate radius or width.
    """
    verts = np.asarray(mesh.vertices)
    y_min = float(verts[:, 1].min())
    y_span = float(verts[:, 1].max() - y_min)
    band = verts[verts[:, 1] < y_min + 0.30 * y_span]
    if len(band) < 8:
        return []
    mx, mz = np.median(band[:, 0]), np.median(band[:, 2])
    names = {(True, True): "Wheel_FL", (True, False): "Wheel_RL",
             (False, True): "Wheel_FR", (False, False): "Wheel_RR"}
    wheels = []
    for (left, front), name in names.items():
        sel = band[(band[:, 0] < mx) == left]
        sel = sel[(sel[:, 2] < mz) == (not front)]
        if len(sel) < 4:
            continue
        # pass 1: lowest quartile of the quadrant is wheel-dominant
        low = sel[sel[:, 1] < np.percentile(sel[:, 1], 60)]
        cy, cz = float(np.mean(low[:, 1])), float(np.mean(low[:, 2]))
        radial = np.sqrt((sel[:, 1] - cy) ** 2 + (sel[:, 2] - cz) ** 2)
        radius = float(np.percentile(radial, 80))
        # pass 2: restrict to verts near the cylinder, re-fit
        near = sel[(radial < radius * 1.1) & (np.abs(sel[:, 0] - np.median(sel[:, 0])) < radius)]
        if len(near) >= 4:
            cy, cz = float(np.mean(near[:, 1])), float(np.mean(near[:, 2]))
            radial = np.sqrt((near[:, 1] - cy) ** 2 + (near[:, 2] - cz) ** 2)
            radius = float(np.percentile(radial, 85))
            x_span = float(np.ptp(near[:, 0]))
            half_width = float(np.clip(x_span / 2.0, radius * 0.2, radius * 0.6))
            center_x = float(np.mean(near[:, 0]))
        else:
            center_x = float(np.mean(sel[:, 0]))
            half_width = radius * 0.35
        wheels.append({
            "name": name,
            "center": [center_x, cy, cz],
            "axis": [1.0, 0.0, 0.0],
            "radius": radius,
            "half_width": half_width,
        })
    return wheels


def partition(mesh: trimesh.Trimesh, wheels: list[dict]):
    """Assign each face to a wheel cylinder or the chassis; cap boundaries."""
    centroids = mesh.triangles_center
    assignment = np.full(mesh.faces.shape[0], -1, dtype=int)
    for wi, wheel in enumerate(wheels):
        c = np.asarray(wheel["center"], dtype=float)
        axis = np.asarray(wheel["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        rel = centroids - c
        along = rel @ axis
        radial_vec = rel - np.outer(along, axis)
        radial = np.linalg.norm(radial_vec, axis=1)
        inside = (radial < wheel["radius"]) & (np.abs(along) < wheel["half_width"])
        assignment[inside] = wi  # later wheels win overlaps

    parts = []
    for wi, wheel in enumerate(wheels):
        faces = mesh.faces[assignment == wi]
        if len(faces) == 0:
            continue
        part = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=faces, process=False)
        part.remove_unreferenced_vertices()
        # carry UVs if the source has them
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None)
        if uv is not None and len(uv) == len(mesh.vertices):
            part.visual = trimesh.visual.TextureVisuals(
                uv=uv[_remap_uv(mesh, part)],
                material=getattr(visual, "material", None),
            )
        part.fill_holes()
        pivot = np.asarray(wheel["center"], dtype=float)
        part.vertices -= pivot
        parts.append((wheel["name"], pivot, part))
    chassis_faces = mesh.faces[assignment == -1]
    chassis = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=chassis_faces, process=False)
    chassis.remove_unreferenced_vertices()
    chassis.fill_holes()
    parts.append(("Chassis", np.zeros(3), chassis))
    return parts


def _remap_uv(mesh: trimesh.Trimesh, part: trimesh.Trimesh):
    """Map part vertices back to source indices for UV lookup (nearest within tol)."""
    _, idx = mesh.kdtree.query(part.vertices)
    return idx


def build_scene(parts) -> trimesh.Scene:
    scene = trimesh.Scene()
    for name, pivot, mesh in parts:
        transform = np.eye(4)
        transform[:3, 3] = pivot
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform)
    return scene


def main() -> int:
    src = open(sys.argv[1], "rb").read()
    out = sys.argv[2] if len(sys.argv) > 2 else "vehicle-rigged.glb"
    mesh = load_single_mesh(src)
    print(f"input: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces, "
          f"bounds {mesh.bounds[0].round(3)} .. {mesh.bounds[1].round(3)}")
    mesh, dropped = strip_ground_plane(mesh)
    print(f"ground strip: removed {dropped} faces -> {mesh.faces.shape[0]} faces, "
          f"bounds {mesh.bounds[0].round(3)} .. {mesh.bounds[1].round(3)}")
    wheels = suggest_wheels(mesh)
    print(f"suggested {len(wheels)} wheels:")
    for w in wheels:
        print(f"  {w['name']}: center={[round(v,3) for v in w['center']]} "
              f"r={w['radius']:.3f} hw={w['half_width']:.3f}")
    parts = partition(mesh, wheels)
    for name, pivot, part in parts:
        print(f"  {name}: {part.faces.shape[0]} faces, pivot={[round(v,3) for v in pivot]}, "
              f"watertight={part.is_watertight}")
    scene = build_scene(parts)
    data = scene.export(file_type="glb")
    open(out, "wb").write(data)
    print(f"wrote {out} ({len(data)} bytes)")

    # verify: reload and check node names + pivots
    check = trimesh.load(out, force="scene")
    print("reloaded nodes:", sorted(check.graph.nodes_geometry))
    for node in check.graph.nodes_geometry:
        if node == "Chassis":
            continue
        xf, geom = check.graph.get(node)
        pos = xf[:3, 3]
        print(f"  {node}: pivot={pos.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
