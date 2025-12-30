import bpy
import os
from dataclasses import dataclass
from .utils import *
from .registry import SceneRegistry
from .node_graph import RMNodeGraph, RMInput, RMNode
from .node import export_node, _handle_image, _export_image
from .materials import _is_black
from .addon_preferences import get_prefs


@dataclass
class EnvironmentSettings:
    texture: str
    brightness: float


def _normalize_color(color):
    brightness = max(color)
    if brightness <= 0:
        return None, 0.0
    normalized = [c / brightness for c in color]
    return normalized, brightness


def _write_constant_environment_texture(registry: SceneRegistry, color):
    normalized, brightness = _normalize_color(color)
    if brightness <= 0:
        return None, 0.0

    suffix = "_".join([f"{c:.4f}".replace(".", "p") for c in normalized])
    image_name = f"NomadEnv_{suffix}"

    image = bpy.data.images.new(
        image_name, width=1, height=1, alpha=False, float_buffer=True)
    image.generated_color = (*normalized, 1.0)
    image.use_generated_float = True
    image.colorspace_settings.name = "Linear"
    try:
        tex_dir_name = get_prefs().tex_dir_name
        os.makedirs(os.path.join(registry.path, tex_dir_name), exist_ok=True)
        texture_path = _handle_image(registry, image)
    finally:
        bpy.data.images.remove(image)

    return texture_path, brightness


def _export_environment_image(registry: SceneRegistry, image: bpy.types.Image):
    tex_dir_name = get_prefs().tex_dir_name
    os.makedirs(os.path.join(registry.path, tex_dir_name), exist_ok=True)
    file_name = bpy.path.clean_name(image.name) + ".exr"
    relative_path = os.path.join(tex_dir_name, file_name)
    _export_image(registry, image, os.path.join(
        registry.path, relative_path), is_f32=True, keep_format=False)
    return relative_path.replace("\\", "/")


def export_world_background(registry: SceneRegistry, scene: bpy.types.Scene):
    if not scene.world:
        return None

    # Export basic world if no shading nodes are given
    if not scene.world.node_tree:
        color = scene.world.color
        if not _is_black(color):
            texture_path, brightness = _write_constant_environment_texture(
                registry, [float(color[0]), float(color[1]), float(color[2])])
            if texture_path:
                return EnvironmentSettings(texture=texture_path, brightness=brightness)
        return None  # No world

    node_graph = RMNodeGraph(
        registry, scene.world.name_full, scene.world.node_tree)
    node_graph.inline_node_groups_recursively()
    node_graph.remove_reroute_nodes()
    node_graph.remove_muted_nodes()
    node_graph.remove_layout_nodes()

    for (node_name, rm_node) in node_graph.nodes.items():
        node = rm_node.bl_node
        if isinstance(node, bpy.types.ShaderNodeOutputWorld):
            if node.is_active_output:
                return _export_world(registry, rm_node.input("Surface"))

    return None  # No active output


def _export_background(registry: SceneRegistry, bsdf_node: RMNode):
    color = bsdf_node.input("Color")
    strength = bsdf_node.input("Strength")

    emission_scale = 1
    if strength.is_linked():
        registry.error(
            "Only constant values for emission strength are supported")
    elif strength.has_value():
        emission_scale = float(strength.value)
    if emission_scale == 0:
        return None

    linked = color.linked_node() if color.is_linked() else None

    if not color.is_linked():
        if not color.has_value() or _is_black(color.value):
            return None
        color_values = [float(c) * emission_scale for c in color.value[0:3]]
        texture_path, brightness = _write_constant_environment_texture(
            registry, color_values)
        if texture_path:
            return EnvironmentSettings(texture=texture_path, brightness=brightness)
        return None

    if linked and isinstance(linked.bl_node, bpy.types.ShaderNodeTexEnvironment):
        image = linked.bl_node.image
        if image:
            try:
                path = _export_environment_image(registry, image)
                if path:
                    return EnvironmentSettings(texture=path, brightness=emission_scale)
            except Exception as ex:
                registry.error(f"Failed to export environment texture {image.name}: {ex}")
        else:
            registry.error(f"Environment texture node {linked.bl_node.name} has no image")
        return None

    exported_texture = export_node(
        registry, color, exposure=emission_scale)
    tex_type = exported_texture.attributes.get("type")

    if tex_type == "image":
        path = exported_texture.attributes.get("filename")
        if path and not path.lower().endswith(".exr"):
            if linked and hasattr(linked.bl_node, "image") and linked.bl_node.image:
                try:
                    path = _export_environment_image(registry, linked.bl_node.image)
                except Exception as ex:
                    registry.error(f"Failed to export environment texture {linked.bl_node.image.name}: {ex}")
            else:
                registry.warn("Environment texture is not EXR; attempting to use original path")
        brightness = float(exported_texture.attributes.get("exposure", 1.0))
        return EnvironmentSettings(texture=path, brightness=brightness)

    if tex_type == "constant":
        value = exported_texture.attributes.get("value")
        if value:
            values = [float(v) for v in value.split(",")]
            texture_path, brightness = _write_constant_environment_texture(
                registry, values)
            if texture_path:
                return EnvironmentSettings(texture=texture_path, brightness=brightness)

    registry.error("Unsupported environment configuration for world background")
    return None


_world_handlers: dict[str, any] = {
    "ShaderNodeEmission": _export_background,
    "ShaderNodeBackground": _export_background,
}


def _export_world(registry: SceneRegistry, input: RMInput):
    if not input.is_linked():
        return None  # black world

    world_node = input.linked_node()

    if world_node is None:
        registry.error(f"Material {input.node_graph.name} has no valid node")
        return None

    for (typename, handler) in _world_handlers.items():
        if hasattr(bpy.types, typename) and isinstance(world_node.bl_node, getattr(bpy.types, typename)):
            return handler(registry, world_node)

    exported_texture = export_node(registry, input)
    tex_type = exported_texture.attributes.get("type")

    if tex_type == "image":
        path = exported_texture.attributes.get("filename")
        return EnvironmentSettings(
            texture=path,
            brightness=float(exported_texture.attributes.get("exposure", 1.0)))

    if tex_type == "constant":
        value = exported_texture.attributes.get("value")
        if value:
            values = [float(v) for v in value.split(",")]
            texture_path, brightness = _write_constant_environment_texture(
                registry, values)
            if texture_path:
                return EnvironmentSettings(texture=texture_path, brightness=brightness)

    registry.error("Unsupported world setup; unable to export environment")
    return None
