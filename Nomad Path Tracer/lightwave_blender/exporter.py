import bpy
import math
import mathutils
import os

from .shape import export_shape
from .utils import str_flat_array, str_float
from .addon_preferences import get_prefs
from .xml_node import XMLNode, XMLRootNode
from .registry import SceneRegistry
from .world import export_world_background
from .node import _handle_image
from .light import export_lights
from .camera import export_camera

_RESIDENCY_PRESETS: dict[str, dict] = {
    "distance": {
        "strategy": "distance",
        "attributes": {
            "textureResidencyMemoryCapMB": 16,
            "lodEnterDistance": 50,
            "lodExitDistance": 75,
            "residencyCooldown": 1,
            "lodToggleBudget": 10000,
        },
    },
    "environment": {
        "strategy": "environment",
        "attributes": {
            "textureResidencyMemoryCapMB": 512,
            "envTargetFraction": 0.78,
            "envEscapeThreshold": 0.18,
            "envMinActive": 150,
            "envToggleBudget": 15000,
            "envDepthRadii": (18, 36, 60),
            "envDepthWeights": (1.4, 1.0, 0.6),
        },
    },
    "predictive": {
        "strategy": "predictive",
        "attributes": {
            "textureResidencyMemoryCapMB": 512,
            "envTargetFraction": 0.72,
            "envEscapeThreshold": 0.16,
            "envMinActive": 150,
            "envToggleBudget": 15000,
            "envDepthRadii": (18, 36, 60),
            "envDepthWeights": (1.4, 1.0, 0.6),
        },
    },
    "probabilistic": {
        "strategy": "probabilistic",
        "attributes": {
            "textureResidencyMemoryCapMB": 512,
            "probabilityDecay": 0.92,
            "probabilityThreshold": 0.42,
            "probabilityMinActive": 150,
            "probabilityToggleBudget": 14000,
        },
    },
    "unified": {
        "strategy": "unified",
        "attributes": {
            "textureResidencyMemoryCapMB": 512,
            "unifiedEnergyWeight": 0.85,
            "unifiedHitWeight": 1.95,
            "unifiedCoverageWeight": 1.55,
            "unifiedDistanceWeight": 0.55,
        },
    },
}


def _material_attributes(registry: SceneRegistry, inst, mat_id: int):
    mat = None
    if mat_id < len(inst.object.material_slots):
        mat = inst.object.material_slots[mat_id].material

    attrs = {
        "albedo": str_flat_array((0.7, 0.7, 0.7)),
        "emission": str_flat_array((0.0, 0.0, 0.0)),
        "materialType": 0,
        "emissionPower": 0,
    }

    if mat is None or not getattr(registry.settings, "export_materials", True):
        return attrs

    if hasattr(mat, "diffuse_color"):
        color = mat.diffuse_color
        attrs["albedo"] = str_flat_array((color[0], color[1], color[2]))

    def _image_from_socket(socket):
        if socket is None or not socket.is_linked:
            return None
        node = socket.links[0].from_node
        if isinstance(node, bpy.types.ShaderNodeTexImage):
            return node.image
        if isinstance(node, bpy.types.ShaderNodeNormalMap):
            return _image_from_socket(node.inputs.get("Color"))
        return None

    if mat.use_nodes and mat.node_tree is not None:
        principled = next((n for n in mat.node_tree.nodes
                           if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled)), None)
        if principled is not None:
            base_color = principled.inputs.get("Base Color")
            if base_color and base_color.default_value is not None:
                attrs["albedo"] = str_flat_array(base_color.default_value[0:3])

            emission = principled.inputs.get("Emission")
            if emission and emission.default_value is not None:
                attrs["emission"] = str_flat_array(emission.default_value[0:3])

            emission_strength = principled.inputs.get("Emission Strength")
            if emission_strength is not None:
                attrs["emissionPower"] = emission_strength.default_value

            diffuse_image = _image_from_socket(base_color)
            if diffuse_image is not None:
                diffuse_path = _handle_image(registry, diffuse_image)
                if diffuse_path:
                    attrs["diffuseTexture"] = diffuse_path

            specular_image = _image_from_socket(principled.inputs.get("Specular"))
            if specular_image is not None:
                specular_path = _handle_image(registry, specular_image)
                if specular_path:
                    attrs["specularTexture"] = specular_path

            normal_image = None
            normal_socket = principled.inputs.get("Normal")
            if normal_socket is not None:
                normal_image = _image_from_socket(normal_socket)
            if normal_image is not None:
                normal_path = _handle_image(registry, normal_image)
                if normal_path:
                    attrs["normalTexture"] = normal_path

    return attrs


def _residency_attributes(settings) -> dict:
    if not getattr(settings, "enable_residency_settings", False):
        return {}

    preset_key = getattr(settings, "residency_preset", "none")
    preset = _RESIDENCY_PRESETS.get(preset_key)
    if preset is None:
        return {}

    attrs: dict = {"residencyStrategy": preset["strategy"]}
    for key, value in preset["attributes"].items():
        if isinstance(value, (tuple, list)):
            attrs[key] = str_flat_array(value)
        else:
            attrs[key] = value
    return attrs

def export_objects(registry: SceneRegistry):
    result: list[XMLNode] = []

    def _is_uniform_scale(scale_vec: mathutils.Vector, epsilon: float = 1.0e-4) -> bool:
        return (abs(scale_vec.x - scale_vec.y) <= epsilon and
                abs(scale_vec.x - scale_vec.z) <= epsilon)

    def _basis_matches_rotation(basis_matrix: mathutils.Matrix,
                                rotation: mathutils.Quaternion,
                                scale_vec: mathutils.Vector,
                                epsilon: float = 1.0e-4) -> bool:
        rot_matrix = rotation.to_matrix().to_3x3()
        expected_basis_x = rot_matrix.col[0] * scale_vec.x
        expected_basis_y = rot_matrix.col[1] * scale_vec.y
        expected_basis_z = rot_matrix.col[2] * scale_vec.z

        def _close(a: mathutils.Vector, b: mathutils.Vector) -> bool:
            return (a - b).length <= epsilon * max(1.0, a.length, b.length)

        return (_close(basis_matrix.col[0], expected_basis_x) and
                _close(basis_matrix.col[1], expected_basis_y) and
                _close(basis_matrix.col[2], expected_basis_z))

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

        shapes: list[XMLNode] = export_shape(registry, object_eval)
        if len(shapes) == 0:
            registry.warn(f"Entity {object_eval.name} has no material or shape and will be ignored")
            continue

        basis = inst.matrix_world.to_3x3()
        basis_x = basis.col[0]
        basis_y = basis.col[1]
        basis_z = basis.col[2]

        location, rotation, scale = inst.matrix_world.decompose()
        rotation_euler = rotation.to_euler('XYZ')
        has_uniform_scale = _is_uniform_scale(scale)
        uses_rotation_attributes = has_uniform_scale and _basis_matches_rotation(
            basis, rotation, scale)
        scale_value = (scale.x + scale.y + scale.z) / 3.0 if uses_rotation_attributes \
            else (basis_x.length + basis_y.length + basis_z.length) / 3.0

        for shape in shapes:
            filename = shape.attributes.get("filename")
            if filename is None:
                continue

            mat_id = shape.attributes.get("material_index", 0)

            mesh_attributes = {
                "file": filename,
                "position": str_flat_array(location),
                "scale": str_float(scale_value),
                "clusterMaxTriangles": 0,
                "clusterMaxExtent": 0.0,
            }

            if uses_rotation_attributes:
                mesh_attributes["rotation"] = str_flat_array((
                    math.degrees(rotation_euler.x),
                    math.degrees(rotation_euler.y),
                    math.degrees(rotation_euler.z)
                ))
            else:
                mesh_attributes["basisX"] = str_flat_array(basis_x)
                mesh_attributes["basisY"] = str_flat_array(basis_y)
                mesh_attributes["basisZ"] = str_flat_array(basis_z)

            mesh_node = XMLNode("Mesh", **mesh_attributes)

            for key, value in _material_attributes(registry, inst, mat_id).items():
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
    })
    scene.attributes.update(_residency_attributes(settings))

    # Create a path for meshes & textures
    rootPath = os.path.dirname(filepath)
    meshDir = os.path.join(rootPath, get_prefs().mesh_dir_name)
    texDir = os.path.join(rootPath, get_prefs().tex_dir_name)
    os.makedirs(meshDir, exist_ok=True)
    os.makedirs(texDir, exist_ok=True)

    registry = SceneRegistry(rootPath, depsgraph, settings, op)

    if settings.enable_camera and depsgraph.scene.camera is not None:
        scene.add_children(export_camera(registry))

    scene.add_children(export_objects(registry))
    scene.add_children(export_lights(registry))

    if settings.enable_background:
        environment = export_world_background(registry, depsgraph.scene)
        if environment:
            scene.attributes["environmentTexture"] = environment.texture
            scene.attributes["environmentBrightness"] = environment.brightness

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
