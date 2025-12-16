import bpy
import mathutils
import os

from .shape import export_shape
from .utils import str_flat_array
from .addon_preferences import get_prefs
from .xml_node import XMLNode, XMLRootNode
from .registry import SceneRegistry
from .world import export_world_background


def _default_material_attributes(inst, mat_id: int):
    mat = None
    if mat_id < len(inst.object.material_slots):
        mat = inst.object.material_slots[mat_id].material

    if mat is not None and hasattr(mat, "diffuse_color"):
        color = mat.diffuse_color
        albedo = (color[0], color[1], color[2])
    else:
        albedo = (0.7, 0.7, 0.7)

    return {
        "albedo": str_flat_array(albedo),
        "emission": str_flat_array((0.0, 0.0, 0.0)),
        "materialType": 0,
        "emissionPower": 0,
    }

def export_objects(registry: SceneRegistry):
    result: list[XMLNode] = []

    for inst in registry.depsgraph.object_instances:
        object_eval = inst.object
        if object_eval is None:
            continue
        if registry.settings.use_selection and not object_eval.original.select_get():
            continue
        if not registry.settings.use_selection and not inst.show_self:
            continue

        if object_eval.type not in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'CURVES'}:
            continue

        shapes: list[XMLNode] = registry.export(
            object_eval.original.data, lambda unique_name: export_shape(registry, object_eval))
        if len(shapes) == 0:
            registry.warn(f"Entity {object_eval.name} has no material or shape and will be ignored")
            continue

        basis = inst.matrix_world.to_3x3()
        basis_x = basis.col[0]
        basis_y = basis.col[1]
        basis_z = basis.col[2]
        location = inst.matrix_world.to_translation()
        rotation = inst.matrix_world.to_euler('XYZ')

        for (mat_id, shape) in enumerate(shapes):
            filename = shape.attributes.get("filename")
            if filename is None:
                continue

            mesh_node = XMLNode(
                "Mesh",
                file=filename,
                position=str_flat_array(location),
                basisX=str_flat_array(basis_x),
                basisY=str_flat_array(basis_y),
                basisZ=str_flat_array(basis_z),
                rotation=str_flat_array((
                    mathutils.rad2deg(rotation.x),
                    mathutils.rad2deg(rotation.y),
                    mathutils.rad2deg(rotation.z)
                )),
                clusterMaxTriangles=0,
                clusterMaxExtent=0.0,
            )

            for key, value in _default_material_attributes(inst, mat_id).items():
                mesh_node.attributes[key] = value

            result.append(mesh_node)

    return result


def export_scene(op, filepath, context, settings):
    depsgraph = context.evaluated_depsgraph_get() if not isinstance(
        context, bpy.types.Depsgraph) else context

    # Root
    root = XMLRootNode()
    scene = XMLNode("Scene")

    render = depsgraph.scene.render
    res_x = int(render.resolution_x * render.resolution_percentage * 0.01)
    res_y = int(render.resolution_y * render.resolution_percentage * 0.01)

    if getattr(depsgraph.scene, 'cycles', None) is not None and bpy.context.engine == 'CYCLES':
        cycles = depsgraph.scene.cycles
        max_depth = cycles.max_bounces + 1
    else:
        max_depth = 8

    scene.attributes.update({
        "width": res_x,
        "height": res_y,
        "maxRayDepth": max_depth,
        "startCompacted": True,
        "residencyStrategy": "distance",
        "textureResidencyMemoryCapMB": 16,
        "lodEnterDistance": 50,
        "lodExitDistance": 75,
        "residencyCooldown": 1,
        "lodToggleBudget": 10000,
    })

    # Create a path for meshes & textures
    rootPath = os.path.dirname(filepath)
    meshDir = os.path.join(rootPath, get_prefs().mesh_dir_name)
    texDir = os.path.join(rootPath, get_prefs().tex_dir_name)
    os.makedirs(meshDir, exist_ok=True)
    os.makedirs(texDir, exist_ok=True)

    registry = SceneRegistry(rootPath, depsgraph, settings, op)

    if settings.enable_camera and depsgraph.scene.camera is not None:
        camera = depsgraph.scene.camera
        cam_matrix = camera.matrix_world
        cam_pos = cam_matrix.to_translation()
        cam_forward = cam_matrix.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
        look_at = cam_pos + cam_forward

        camera_path = XMLNode("CameraPath")
        camera_path.add("Keyframe", frame=depsgraph.scene.frame_current,
                        position=str_flat_array(cam_pos),
                        lookAt=str_flat_array(look_at))
        scene.add_child(camera_path)

    scene.add_children(export_objects(registry))

    if settings.enable_background:
        scene.add_children(export_world_background(registry, depsgraph.scene))

    root.add_child(scene)

    # Remove mesh & texture directory if empty
    try:
        if len(os.listdir(meshDir)) == 0:
            os.rmdir(meshDir)
        if len(os.listdir(texDir)) == 0:
            os.rmdir(texDir)
    except:
        pass  # Ignore any errors

    return root


def export_scene_to_file(op, filepath, context, settings):
    root = export_scene(op, filepath, context, settings)

    # Write the result into a file
    with open(filepath, 'w') as fp:
        fp.write(root.dump())
