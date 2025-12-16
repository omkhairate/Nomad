import os
import bmesh

from .addon_preferences import get_prefs
from .registry import SceneRegistry
from .xml_node import XMLNode


def get_shape_name_base(obj, inst):
    modifiers = [mod.type for mod in obj.original.modifiers]
    has_nodes = 'NODES' in modifiers

    if has_nodes:
        # Not sure how to ensure shapes with nodes are handled as uniques
        # TODO: We better join them by material
        id = hex(inst.random_id).replace("0x", "").replace('-', 'M').upper()
        return f"{obj.name}_{id}"

    try:
        return f"{obj.data.name}-shape"
    except:
        return f"{obj.original.data.name}-shape"  # We use the original mesh name!


def _shape_name_material(name, mat_id):
    return f"_m_{mat_id}_{name}"

def _write_obj(filepath, bm: bmesh.types.BMesh):
    uv_layer = bm.loops.layers.uv.active

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Exported by Lightwave Blender addon\n")

        for v in bm.verts:
            f.write(f"v {v.co.x} {v.co.y} {v.co.z}\n")

        vt_lookup = {}
        next_vt_index = 1
        if uv_layer is not None:
            for face in bm.faces:
                for loop in face.loops:
                    uv = loop[uv_layer].uv
                    vt_lookup[(face.index, loop.index)] = next_vt_index
                    f.write(f"vt {uv.x} {uv.y}\n")
                    next_vt_index += 1

        for v in bm.verts:
            n = v.normal
            f.write(f"vn {n.x} {n.y} {n.z}\n")

        for face in bm.faces:
            parts = []
            for loop in face.loops:
                v_idx = loop.vert.index + 1
                vt_idx = vt_lookup.get((face.index, loop.index))
                vn_idx = loop.vert.index + 1
                if uv_layer is not None:
                    parts.append(f"{v_idx}/{vt_idx}/{vn_idx}")
                else:
                    parts.append(f"{v_idx}//{vn_idx}")
            f.write("f " + " ".join(parts) + "\n")


def _export_bmesh_by_material(registry: SceneRegistry, me) -> list[XMLNode]:
    mat_count = len(me.materials)
    shapes = []

    def _export_for_mat(mat_id, abs_filepath):
        bm = bmesh.new()
        bm.from_mesh(me)

        if mat_count > 1:
            for f in list(bm.faces):
                if f.material_index != mat_id and not ((f.material_index < 0 or f.material_index >= mat_count) and mat_id == mat_count - 1):
                    bm.faces.remove(f)

        if len(bm.verts) == 0 or len(bm.faces) == 0 or not bm.is_valid:
            bm.free()
            return False

        bmesh.ops.connect_verts_concave(bm, faces=bm.faces)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.normal_update()

        os.makedirs(os.path.dirname(abs_filepath), exist_ok=True)
        _write_obj(abs_filepath, bm)

        bm.free()
        return True

    if mat_count == 0:
        mat_count = 1

    for mat_id in range(0, mat_count):
        shape_name = me.name if mat_count <= 1 else _shape_name_material(me.name, mat_id)
        rel_filepath = os.path.join(get_prefs().mesh_dir_name, shape_name + ".obj")
        abs_filepath = os.path.join(registry.path, rel_filepath)

        if os.path.exists(abs_filepath) and not registry.settings.overwrite_existing_meshes:
            pass
        elif _export_for_mat(mat_id, abs_filepath):
            pass
        else:
            continue

        shapes.append(XMLNode("shape", filename=rel_filepath.replace('\\', '/'), material_index=mat_id))

    return shapes


def export_shape(registry: SceneRegistry, obj) -> list[XMLNode]:
    # TODO: We want the mesh to be evaluated with renderer (or viewer) depending on user input
    # This is not possible currently, as access to `mesh_get_eval_final` (COLLADA) is not available
    # nor is it possible to setup via dependency graph, see https://devtalk.blender.org/t/get-render-dependency-graph/12164
    try:
        me = obj.to_mesh(preserve_all_data_layers=False, depsgraph=registry.depsgraph)
    except RuntimeError as e:
        registry.error(f"Could not convert to mesh: {str(e)}")
        return []

    shapes = _export_bmesh_by_material(registry, me)
    obj.to_mesh_clear()

    return shapes
