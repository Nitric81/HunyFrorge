"""HunyForge Unreal Engine package setup.

Runs inside the Unreal Editor Python interpreter (Python Editor Script
Plugin). The script self-probes the running engine for the APIs this
package needs, performs every step it can, and writes a JSON report
describing what was automated and what remains manual.

Usage (headless):
    UnrealEditor-Cmd.exe "<project>.uproject" ^
        -ExecutePythonScript="<this file>" -unattended -nop4 -nosplash

Or run from the Editor's Python console / Output Log Python REPL.

Environment variables:
    HUNYFORGE_SOURCE_DIR    Folder containing the GLB artifacts and
                            hunyforge-unreal-manifest.json.
                            Default: this script's directory.
    HUNYFORGE_CONTENT_PATH  Content destination for imported assets.
                            Default: /Game/HunyForge
    HUNYFORGE_REPORT_PATH   JSON result report path.
                            Default: <source>/hunyforge-unreal-result.json
    HUNYFORGE_PROBE_ONLY    "1" = probe capabilities and verify a demo
                            GLB import only; do not import the package.
"""

import json
import os
import struct
import tempfile
from pathlib import Path

import unreal

REPORT_VERSION = 1

# Candidate API names, probed at runtime so the same script adapts to
# whatever the installed engine exposes.
LOD_INSERT_CANDIDATES = ("import_lod", "add_lod_from_file")
LOD_AUTO_CANDIDATES = ("set_lod_reduction_settings", "set_lod_build_settings")
SIMPLE_COLLISION_CANDIDATES = ("add_simple_collisions", "add_simple_collisions_with_notification")
CONVEX_COLLISION_CANDIDATES = ("set_convex_decomposition_collisions", "set_convex_decomposition_collisions_with_notification")
COLLISION_CHECK_CANDIDATES = ("has_simple_collisions", "get_simple_collision_count", "has_collision")
COLLISION_ENUM_CANDIDATES = ("ScriptingCollisionShapeType", "CollisionShape", "ScriptingCollisionShape")


def _log(message: str) -> None:
    unreal.log(f"[HunyForge] {message}")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _engine_version() -> str:
    try:
        return unreal.SystemLibrary.get_engine_version()
    except Exception:
        return "unknown"


def _has(cls, name: str) -> bool:
    return cls is not None and callable(getattr(cls, name, None))


def _collision_enum(member_hint: str):
    for name in COLLISION_ENUM_CANDIDATES:
        enum_type = getattr(unreal, name, None)
        if enum_type is None:
            continue
        for member in dir(enum_type):
            if member_hint in member.lower():
                try:
                    return getattr(enum_type, member)
                except Exception:
                    continue
    return None


def _demo_glb_bytes() -> bytes:
    """Minimal valid triangle GLB used to verify .glb import works."""
    positions = struct.pack("<9f", -0.8, 0, 0, 0.8, 0, 0, 0, 1.4, 0)
    indices = struct.pack("<3H", 0, 1, 2) + b"\x00\x00"
    binary = positions + indices
    document = {
        "asset": {"version": "2.0", "generator": "HunyForge probe"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": 6, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [-0.8, 0, 0], "max": [0.8, 1.4, 0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    return (
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(encoded) + 8 + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _asset_tools():
    try:
        return unreal.AssetToolsHelpers.get_asset_tools()
    except Exception:
        return None


def _import_file(path: Path, destination: str) -> list[str]:
    """Import one source file via the engine's registered factory for its
    extension. Returns imported asset object paths."""
    tools = _asset_tools()
    if tools is None:
        raise RuntimeError("AssetTools is unavailable in this editor build")
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(path))
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    tools.import_asset_tasks([task])
    paths = []
    for attr in ("imported_object_paths", "imported_object_path_names"):
        try:
            value = task.get_editor_property(attr)
            if value:
                paths = [str(item) for item in value]
                break
        except Exception:
            continue
    if not paths and hasattr(task, "result"):
        try:
            result = task.result
            paths = [obj.get_path_name() for obj in getattr(result, "imported_objects", []) or []]
        except Exception:
            pass
    return paths


def _probe_demo_import(destination: str) -> dict:
    """Verify the engine can actually import a .glb, not just that an
    importer plugin is enabled."""
    result = {"attempted": True, "ok": False, "detail": ""}
    tmp = Path(tempfile.mkdtemp(prefix="hunyforge-probe-")) / "probe.glb"
    try:
        tmp.write_bytes(_demo_glb_bytes())
        paths = _import_file(tmp, destination)
        if not paths:
            result["detail"] = "no importer accepted the .glb file"
            return result
        mesh = None
        for asset_path in paths:
            try:
                loaded = unreal.EditorAssetLibrary.load_asset(asset_path.split(".")[0])
            except Exception:
                loaded = None
            if isinstance(loaded, unreal.StaticMesh):
                mesh = loaded
                break
        if mesh is None:
            result["detail"] = f"imported assets were not StaticMesh: {paths}"
        else:
            result["ok"] = True
            result["detail"] = f"imported {paths[0]}"
        for asset_path in paths:
            try:
                unreal.EditorAssetLibrary.delete_asset(asset_path.split(".")[0])
            except Exception:
                pass
        return result
    except Exception as exc:
        result["detail"] = str(exc)
        return result


def probe_capabilities() -> dict:
    esml = getattr(unreal, "EditorStaticMeshLibrary", None)
    static_mesh_cls = getattr(unreal, "StaticMesh", None)
    return {
        "engine_version": _engine_version(),
        "asset_tools": _asset_tools() is not None,
        "interchange_manager": hasattr(unreal, "InterchangeManager"),
        "material_editing": hasattr(unreal, "MaterialEditingLibrary"),
        "sphyl_elem": hasattr(unreal, "KSphylElem"),
        "lod_insert_api": next((name for name in LOD_INSERT_CANDIDATES if _has(static_mesh_cls, name) or _has(esml, name)), None),
        "lod_auto_api": next((name for name in LOD_AUTO_CANDIDATES if _has(esml, name)), None),
        "simple_collision_api": next((name for name in SIMPLE_COLLISION_CANDIDATES if _has(esml, name)), None),
        "convex_collision_api": next((name for name in CONVEX_COLLISION_CANDIDATES if _has(esml, name)), None),
        "collision_check_api": next((name for name in COLLISION_CHECK_CANDIDATES if _has(esml, name)), None),
        "box_collision_enum": _collision_enum("box") is not None,
    }


def _find_sources(source_dir: Path) -> dict:
    manifest_path = source_dir / "hunyforge-unreal-manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    files = {"manifest": manifest, "lod0": None, "lod1": None, "collision": None}
    glbs = sorted(source_dir.glob("*.glb"))
    def pick(*needles):
        for glb in glbs:
            name = glb.name.lower()
            if any(needle in name for needle in needles):
                return glb
        return None
    files["lod1"] = pick("lod1", "_lod1")
    files["collision"] = pick("collision", "ucx_", "ubx_")
    # "sm_" would false-match "Collision_SM_*" (sorted first), so prefer explicit
    # LOD0 names and otherwise take the first GLB not already claimed above.
    files["lod0"] = pick("lod0", "_lod0", "textured", "rigged") or next(
        (glb for glb in glbs if glb not in (files["lod1"], files["collision"])),
        glbs[0] if glbs else None,
    )
    pngs = sorted(source_dir.glob("*.png"))
    files["leaf_atlas"] = next((p for p in pngs if "leaf" in p.name.lower() or "atlas" in p.name.lower()), None)
    return files


def _apply_lod(mesh, lod1_path: Path, caps: dict, report: dict) -> None:
    esml = getattr(unreal, "EditorStaticMeshLibrary", None)
    api = caps.get("lod_insert_api")
    if api:
        try:
            if hasattr(mesh, api):
                getattr(mesh, api)(1, str(lod1_path))
            else:
                getattr(esml, api)(mesh, 1, str(lod1_path))
            report["actions"].append(f"lod1-imported-via-{api}")
            return
        except Exception as exc:
            report["warnings"].append(f"LOD import via {api} failed: {exc}")
    auto_api = caps.get("lod_auto_api")
    if auto_api:
        try:
            settings_type = getattr(unreal, "StaticMeshReductionSettings", None) or getattr(unreal, "EditorStaticMeshReductionSettings", None)
            if settings_type is not None:
                settings = settings_type()
                for attr, value in (("percent_triangles", 0.5), ("percent_vertices", 0.5)):
                    try:
                        settings.set_editor_property(attr, value)
                        break
                    except Exception:
                        continue
                getattr(esml, auto_api)(mesh, 1, settings)
                report["actions"].append(f"lod1-auto-generated-via-{auto_api}")
                report["warnings"].append("LOD1 was auto-generated by the engine (source LOD1 file could not be inserted); review quality")
                return
        except Exception as exc:
            report["warnings"].append(f"Auto LOD via {auto_api} failed: {exc}")
    report["manual_steps"].append(
        f"Open the imported StaticMesh and add LOD1 manually: LOD Settings > Import LOD > '{lod1_path.name}', or enable automatic LOD generation."
    )


def _apply_collision(mesh, collision_path: Path | None, mode: str, caps: dict, report: dict) -> None:
    esml = getattr(unreal, "EditorStaticMeshLibrary", None)
    if mode == "box":
        enum_value = _collision_enum("box")
        api = caps.get("simple_collision_api")
        if api and enum_value is not None:
            try:
                getattr(esml, api)(mesh, enum_value)
                report["actions"].append(f"box-collision-via-{api}")
                return
            except Exception as exc:
                report["warnings"].append(f"Box collision via {api} failed: {exc}")
        report["manual_steps"].append("Add a box collision in the StaticMesh Editor: Collision > Add Box Simplified Collision.")
        return
    api = caps.get("convex_collision_api")
    if api:
        try:
            getattr(esml, api)(mesh, 1, 32, 100000)
            report["actions"].append(f"convex-collision-via-{api}")
            return
        except TypeError:
            try:
                getattr(esml, api)(mesh)
                report["actions"].append(f"convex-collision-via-{api}")
                return
            except Exception as exc:
                report["warnings"].append(f"Convex collision via {api} failed: {exc}")
        except Exception as exc:
            report["warnings"].append(f"Convex collision via {api} failed: {exc}")
    hint = f"import '{collision_path.name}' and assign it as custom collision" if collision_path else "use Collision > Auto Convex Collision"
    report["manual_steps"].append(f"Add collision in the StaticMesh Editor: {hint}.")


def _verify_mesh(mesh, report: dict) -> None:
    esml = getattr(unreal, "EditorStaticMeshLibrary", None)
    verification = {}
    try:
        verification["lod_count"] = esml.get_lod_count(mesh) if _has(esml, "get_lod_count") else None
    except Exception:
        verification["lod_count"] = None
    verts = {}
    for index in range(0, (verification.get("lod_count") or 1)):
        try:
            verts[f"lod{index}"] = esml.get_number_verts(mesh, index) if _has(esml, "get_number_verts") else None
        except Exception:
            verts[f"lod{index}"] = None
    verification["verts"] = verts
    check = report["capabilities"].get("collision_check_api")
    if check:
        try:
            verification["collision"] = getattr(esml, check)(mesh)
        except Exception:
            verification["collision"] = None
    report["verification"] = verification


def _mesh_materials(mesh) -> list:
    materials = []
    for prop in ("static_materials", "materials"):
        try:
            slots = mesh.get_editor_property(prop) or []
        except Exception:
            continue
        for slot in slots:
            material = getattr(slot, "material_interface", None)
            if isinstance(material, unreal.Material) and material not in materials:
                materials.append(material)
    return materials


def _wind_wpo_chain(material, mel) -> None:
    """Manual sway on World Position Offset, driven by COLOR_0 wind weights.

    sin(time * speed + worldpos.x * freq) * VertexColor.R * amplitude * direction.
    UE 5.8 exposes no SimpleGrassWind Python class, so the graph is built
    from core expressions."""
    def const(material, mel, x, y, value):
        node = mel.create_material_expression(material, unreal.MaterialExpressionConstant, x, y)
        node.set_editor_property("r", value)
        return node

    time_node = mel.create_material_expression(material, unreal.MaterialExpressionTime, -1400, 500)
    world = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -1400, 700)
    freq_vec = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -1200, 850)
    try:
        freq_vec.set_editor_property("constant", unreal.LinearColor(0.03, 0.017, 0.0, 1.0))
    except Exception:
        pass
    phase_x = mel.create_material_expression(material, unreal.MaterialExpressionDotProduct, -1000, 700)
    mel.connect_material_expressions(world, "", phase_x, "A")
    mel.connect_material_expressions(freq_vec, "", phase_x, "B")
    speed = const(material, mel, -1200, 550, 1.3)
    phase_t = mel.create_material_expression(material, unreal.MaterialExpressionMultiply, -1000, 500)
    mel.connect_material_expressions(time_node, "", phase_t, "A")
    mel.connect_material_expressions(speed, "", phase_t, "B")
    add = mel.create_material_expression(material, unreal.MaterialExpressionAdd, -800, 600)
    mel.connect_material_expressions(phase_t, "", add, "A")
    mel.connect_material_expressions(phase_x, "", add, "B")
    sine = mel.create_material_expression(material, unreal.MaterialExpressionSine, -600, 600)
    mel.connect_material_expressions(add, "", sine, "")
    vc = mel.create_material_expression(material, unreal.MaterialExpressionVertexColor, -600, 820)
    sway = mel.create_material_expression(material, unreal.MaterialExpressionMultiply, -400, 660)
    mel.connect_material_expressions(sine, "", sway, "A")
    mel.connect_material_expressions(vc, "R", sway, "B")
    amplitude = const(material, mel, -400, 860, 6.0)
    scaled = mel.create_material_expression(material, unreal.MaterialExpressionMultiply, -250, 700)
    mel.connect_material_expressions(sway, "", scaled, "A")
    mel.connect_material_expressions(amplitude, "", scaled, "B")
    direction = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -250, 900)
    try:
        direction.set_editor_property("constant", unreal.LinearColor(0.7, 0.45, 0.12, 1.0))
    except Exception:
        pass
    offset = mel.create_material_expression(material, unreal.MaterialExpressionMultiply, -60, 660)
    mel.connect_material_expressions(scaled, "", offset, "A")
    mel.connect_material_expressions(direction, "", offset, "B")
    mel.connect_material_property(offset, "", unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)


def _foliage_material_fixup(material, mel, foliage: dict, report: dict) -> None:
    """Two-sided foliage shading + vertex-color-weighted grass wind on WPO."""
    try:
        material.set_editor_property("two_sided", True)
    except Exception:
        pass
    shading = getattr(unreal, "MaterialShadingModel", None)
    for member in ("MSM_TWO_SIDED_FOLIAGE", "MSM_TWOSIDEDFOLIAGE", "TWO_SIDED_FOLIAGE"):
        if shading is not None and hasattr(shading, member):
            try:
                material.set_editor_property("shading_model", getattr(shading, member))
                break
            except Exception:
                continue
    # Keep the imported blend mode (opaque) — PBR texture alpha is not leaf
    # alpha, and a masked blend would clip the solid crown.
    if foliage.get("wind_vertex_colors"):
        _wind_wpo_chain(material, mel)
        report["actions"].append(f"wind-wpo-vertex-color-{material.get_name()}")
    try:
        mel.recompile_material(material)
    except Exception:
        pass
    report["actions"].append(f"foliage-material-{material.get_name()}")


def _mesh_bounds(mesh):
    """Min/max corners of the render bounds, probed across API variants."""
    esml = getattr(unreal, "EditorStaticMeshLibrary", None)
    candidates = [
        lambda: mesh.get_bounding_box(),
        lambda: esml.get_lod_bounding_box(mesh, 0) if esml is not None else None,
        lambda: esml.get_bounding_box(mesh) if esml is not None else None,
        lambda: mesh.get_bounds(),
    ]
    for getter in candidates:
        try:
            bounds = getter()
        except Exception:
            continue
        if bounds is None:
            continue
        if hasattr(bounds, "min") and hasattr(bounds, "max"):
            return bounds.min, bounds.max
        if hasattr(bounds, "origin") and hasattr(bounds, "box_extent"):
            origin, extent = bounds.origin, bounds.box_extent
            return origin - extent, origin + extent
    return None


def _find_imported_texture(content_path: str, skip_name: str = ""):
    """First Texture2D under the package folder, preferring the mesh base color
    over the leaf atlas."""
    first = None
    try:
        assets = unreal.EditorAssetLibrary.list_assets(content_path, recursive=True)
    except Exception:
        return None
    for path in assets:
        try:
            obj = unreal.EditorAssetLibrary.load_asset(path.split(".")[0])
        except Exception:
            continue
        if not isinstance(obj, unreal.Texture2D):
            continue
        if skip_name and skip_name.lower() in path.lower():
            if first is None:
                first = obj
            continue
        return obj
    return first


def _assign_material_slots(mesh, material, report: dict) -> bool:
    """Point every static material slot at the new foliage material."""
    try:
        slot_count = len(mesh.get_editor_property("static_materials") or [])
    except Exception:
        slot_count = 1
    assigned = 0
    for index in range(max(slot_count, 1)):
        try:
            mesh.set_material(index, material)
            assigned += 1
        except Exception:
            break
    if assigned:
        report["actions"].append(f"material-slots-assigned-{assigned}-{material.get_name()}")
        return True
    return False


def _create_foliage_material(mesh, foliage: dict, atlas_texture, content_path: str, caps: dict, report: dict):
    """Build M_<mesh>_Foliage from scratch: base color texture + masked
    two-sided foliage shading + vertex-color-weighted wind on WPO."""
    mel = getattr(unreal, "MaterialEditingLibrary", None)
    tools = _asset_tools()
    factory_type = getattr(unreal, "MaterialFactoryNew", None)
    if mel is None or tools is None or factory_type is None:
        return None
    # Search the mesh's own folder (e.g. /Game/HunyForge/SM_xxxx) so the
    # material samples this mesh's textures, not another import's.
    mesh_path = mesh.get_path_name().split(".")[0]
    mesh_root = mesh_path.rsplit("/", 2)[0] if mesh_path.count("/") >= 3 else content_path
    texture = _find_imported_texture(mesh_root, skip_name="leaf") or _find_imported_texture(content_path, skip_name="leaf") or atlas_texture
    material = None
    material_name = f"M_{mesh.get_name()}_Foliage"
    try:
        material = tools.create_asset(material_name, content_path, unreal.Material, factory_type())
    except Exception:
        pass
    if material is None:
        try:
            material = unreal.EditorAssetLibrary.load_asset(f"{content_path}/{material_name}.{material_name}")
        except Exception as exc:
            report["warnings"].append(f"Foliage material asset creation failed: {exc}")
            return None
    try:
        mel.delete_all_material_expressions(material)
        try:
            mel.disconnect_material_property(material, unreal.MaterialProperty.MP_OPACITY_MASK)
        except Exception:
            pass
        blend = getattr(unreal, "BlendMode", None)
        if blend is not None and hasattr(blend, "BLEND_OPAQUE"):
            try:
                material.set_editor_property("blend_mode", blend.BLEND_OPAQUE)
            except Exception:
                pass
        if texture is not None:
            tex = mel.create_material_expression(material, unreal.MaterialExpressionTextureSample, -900, -100)
            tex.set_editor_property("texture", texture)
            mel.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
        rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -900, 250)
        rough.set_editor_property("r", 0.85)
        mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
        if foliage.get("wind_vertex_colors"):
            _wind_wpo_chain(material, mel)
            report["actions"].append(f"wind-wpo-vertex-color-{material.get_name()}")
        try:
            material.set_editor_property("two_sided", True)
        except Exception:
            pass
        shading = getattr(unreal, "MaterialShadingModel", None)
        for member in ("MSM_TWO_SIDED_FOLIAGE", "MSM_TWOSIDEDFOLIAGE", "TWO_SIDED_FOLIAGE"):
            if shading is not None and hasattr(shading, member):
                try:
                    material.set_editor_property("shading_model", getattr(shading, member))
                    break
                except Exception:
                    continue
        # Solid generated meshes stay opaque — PBR texture alpha is not leaf
        # alpha, so a masked blend would clip the whole crown. The leaf atlas
        # ships separately for card-based foliage.
        mel.recompile_material(material)
        unreal.EditorAssetLibrary.save_loaded_asset(material)
        report["actions"].append(f"foliage-material-created-{material.get_name()}")
    except Exception as exc:
        report["warnings"].append(f"Foliage material graph build failed: {exc}")
        return material
    if _assign_material_slots(mesh, material, report):
        report["actions"].append("foliage-material-assigned-to-slots")
    else:
        report["manual_steps"].append(f"Assign {material.get_path_name().split('.')[0]} to the mesh's material slots in the StaticMesh Editor.")
    return material


def _apply_trunk_capsule(mesh, capsule: dict | None, caps: dict, report: dict) -> None:
    """UE 5.8 Python cannot write AggGeom elements, so this fits a capsule to
    the whole mesh and records the trunk-only dims for a quick manual resize."""
    esml = getattr(unreal, "EditorStaticMeshLibrary", None)
    enum_value = _collision_enum("capsule")
    api = caps.get("simple_collision_api")
    if api and enum_value is not None and esml is not None:
        try:
            getattr(esml, api)(mesh, enum_value)
            report["actions"].append(f"capsule-collision-via-{api}")
            if capsule:
                radius_cm = float(capsule.get("radius") or 0.0) * 100.0
                height_cm = float(capsule.get("top_fraction") or 0.0) * float(capsule.get("height") or 0.0) * 100.0
                report["warnings"].append(
                    "Fitted capsule covers the whole mesh. For trunk-only collision, resize the capsule to "
                    f"radius ~{radius_cm:.0f} cm and height ~{height_cm:.0f} cm in the StaticMesh Editor."
                )
            return
        except Exception as exc:
            report["warnings"].append(f"Capsule collision via {api} failed: {exc}")
    report["manual_steps"].append("Vegetation: add a capsule collision around the trunk only (StaticMesh Editor > Collision > Add Capsule Simplified Collision, then resize to the trunk).")


def _apply_foliage(mesh, sources: dict, manifest: dict, content_path: str, caps: dict, report: dict) -> None:
    foliage = manifest.get("foliage") or {}
    if not foliage:
        return
    atlas_path = sources.get("leaf_atlas")
    atlas_texture = None
    if atlas_path is not None:
        try:
            imported = _import_file(atlas_path, content_path)
            report["actions"].append(f"imported-{atlas_path.name}")
            for path in imported:
                try:
                    loaded = unreal.EditorAssetLibrary.load_asset(path.split(".")[0])
                except Exception:
                    loaded = None
                if isinstance(loaded, unreal.Texture2D):
                    atlas_texture = loaded
                    break
        except Exception as exc:
            report["warnings"].append(f"Leaf atlas import failed: {exc}")
    mel = getattr(unreal, "MaterialEditingLibrary", None)
    materials = _mesh_materials(mesh)
    if mel is not None and materials:
        for material in materials:
            try:
                _foliage_material_fixup(material, mel, foliage, report)
            except Exception as exc:
                report["warnings"].append(f"Foliage material fixup failed on {material.get_name()}: {exc}")
    else:
        material = _create_foliage_material(mesh, foliage, atlas_texture, content_path, caps, report)
        if material is None:
            report["manual_steps"].append(
                "Vegetation material: create a masked two-sided material with the Two Sided Foliage shading model, "
                "assign the imported base color texture, add VertexColor.R x SimpleGrassWind into World Position "
                "Offset for wind sway, and assign it to the mesh's material slots."
            )
    _apply_trunk_capsule(mesh, foliage.get("trunk_capsule"), caps, report)


def _vehicle_steps(manifest: dict, report: dict) -> None:
    vehicle = manifest.get("vehicle")
    if not vehicle:
        return
    wheels = vehicle.get("wheels") or []
    suggested = vehicle.get("suggested") or {}
    report["vehicle"] = {"wheels": len(wheels), "manifest_block_present": True}
    report["manual_steps"].append(
        "Vehicle rig: this package preserves the Chassis + Wheel_* node pivots. "
        "Chaos Vehicles requires a SkeletalMesh with wheel bones; set up a "
        "ChaosWheeledVehiclePawn (or your vehicle pawn) and create one wheel "
        "entry per manifest wheel using the recorded pivots/radii. "
        f"Suggested starting values: suspension_distance={suggested.get('suspension_distance', 0.1)} m, "
        f"wheel_mass={suggested.get('wheel_mass', 20.0)} kg, chassis_mass={suggested.get('chassis_mass', 1200.0)} kg."
    )


def main() -> dict:
    source_dir = Path(_env("HUNYFORGE_SOURCE_DIR", str(Path(__file__).resolve().parent)))
    content_path = _env("HUNYFORGE_CONTENT_PATH", "/Game/HunyForge")
    report_path = Path(_env("HUNYFORGE_REPORT_PATH", str(source_dir / "hunyforge-unreal-result.json")))
    probe_only = _env("HUNYFORGE_PROBE_ONLY", "0") == "1"

    report = {
        "report_version": REPORT_VERSION,
        "ok": False,
        "engine_version": "unknown",
        "capabilities": {},
        "demo_import": {},
        "imported": [],
        "actions": [],
        "warnings": [],
        "manual_steps": [],
        "verification": {},
    }
    try:
        caps = probe_capabilities()
        report["capabilities"] = caps
        report["engine_version"] = caps.get("engine_version", "unknown")
        _log(f"capabilities: {json.dumps(caps)}")
        if not caps.get("asset_tools"):
            report["manual_steps"].append("Editor scripting is unavailable; enable the Python Editor Script Plugin and Editor Scripting Utilities, then rerun.")
            return report
        report["demo_import"] = _probe_demo_import(content_path + "/Probe")
        if probe_only:
            report["ok"] = report["demo_import"].get("ok", False)
            return report
        if not report["demo_import"].get("ok"):
            report["manual_steps"].append(
                "No .glb importer accepted the probe file. Enable Epic's 'glTF Importer' plugin "
                "(Edit > Plugins > Importers) or a supported Interchange glTF pipeline, restart the editor, then rerun."
            )
            return report

        sources = _find_sources(source_dir)
        manifest = sources["manifest"]
        if sources["lod0"] is None:
            report["manual_steps"].append(f"No .glb mesh found in {source_dir}; extract the HunyForge Unreal package first.")
            return report

        imported_paths = _import_file(sources["lod0"], content_path)
        if not imported_paths:
            report["manual_steps"].append(f"Import of {sources['lod0'].name} produced no assets; check the Output Log for importer errors.")
            return report
        report["imported"] = imported_paths
        report["actions"].append(f"imported-{sources['lod0'].name}")

        mesh = None
        for asset_path in imported_paths:
            try:
                loaded = unreal.EditorAssetLibrary.load_asset(asset_path.split(".")[0])
            except Exception:
                loaded = None
            if isinstance(loaded, unreal.StaticMesh):
                mesh = loaded
                break
        if mesh is None:
            report["manual_steps"].append("Imported assets did not include a StaticMesh; inspect the Content Browser import result.")
            return report

        if sources["lod1"] is not None:
            _apply_lod(mesh, sources["lod1"], caps, report)
        collision_mode = str(manifest.get("collision_mode") or "convex_hull")
        if collision_mode != "trunk" and (sources["collision"] is not None or manifest.get("collision")):
            _apply_collision(mesh, sources["collision"], collision_mode, caps, report)
        _apply_foliage(mesh, sources, manifest, content_path, caps, report)
        try:
            unreal.EditorAssetLibrary.save_loaded_asset(mesh)
        except Exception as exc:
            report["warnings"].append(f"Saving the StaticMesh failed: {exc}")
        _vehicle_steps(manifest, report)
        _verify_mesh(mesh, report)
        report["ok"] = True
        report["fully_automated"] = not report["manual_steps"]
        return report
    finally:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _log(f"report written to {report_path}")
        except Exception as exc:
            _log(f"could not write report: {exc}")


# UE executes -ExecutePythonScript files in a context where __name__ is
# not guaranteed to be "__main__", so main() is invoked unconditionally.
# The editor is only quit when the host bootstrap requests it (headless
# runs), so pasting this file into a live editor's console is safe.
try:
    main()
finally:
    if _env("HUNYFORGE_QUIT_EDITOR", "0") == "1":
        try:
            unreal.SystemLibrary.quit_editor()
        except Exception:
            pass
