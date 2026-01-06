import bpy
import math

from .utils import *
from .registry import SceneRegistry
from .xml_node import XMLNode


def _emissive_attributes(emission):
    return {
        "albedo": str_flat_array((0.0, 0.0, 0.0)),
        "emission": str_flat_array(emission),
        "materialType": 0,
        "emissionPower": 1.0,
    }


def _build_point_light(registry: SceneRegistry, light: bpy.types.Light, matrix_world):
    radius = max(light.shadow_soft_size, 1e-3)
    normalization = 1.0 / (4.0 * (math.pi * radius) ** 2)

    emission = [
        normalization * light.energy * light.color[chan]
        for chan in range(3)
    ]

    position = matrix_world.translation

    return XMLNode(
        "Sphere",
        position=str_flat_array(position),
        radius=str_float(radius),
        **_emissive_attributes(emission),
    )


def _area_light_extents(registry: SceneRegistry, light: bpy.types.Light):
    scale_x = light.size
    if light.shape in {"SQUARE", "DISK"}:
        scale_y = light.size
    elif light.shape in {"RECTANGLE", "ELLIPSE"}:
        scale_y = light.size_y
    else:
        registry.warn(f"Unsupported light shape '{light.shape}'")
        scale_y = light.size
    return scale_x, scale_y


def _build_rectangle_light(registry: SceneRegistry, light: bpy.types.Light, inst_name: str, matrix_world, *, scale_x: float, scale_y: float):
    basis = matrix_world.to_3x3()
    u = basis.col[0] * (scale_x * 0.5)
    v = basis.col[1] * (scale_y * 0.5)

    lensqr_x = u.length_squared
    lensqr_y = v.length_squared

    if lensqr_x <= 0.0 or lensqr_y <= 0.0:
        registry.warn(f"Light '{inst_name}' has zero area and will be skipped")
        return None

    normalization = 1.0 / (16.0 * (lensqr_x * lensqr_y) ** 0.5)
    emission = [
        normalization * light.energy * light.color[chan]
        for chan in range(3)
    ]

    return XMLNode(
        "Rectangle",
        position=str_flat_array(matrix_world.translation),
        u=str_flat_array(u),
        v=str_flat_array(v),
        **_emissive_attributes(emission),
    )


def _sun_extents(light: bpy.types.Light):
    half_size = math.tan(getattr(light, "angle", 0.0) * 0.5)
    return max(half_size * 2.0, 1e-3)


def export_light(registry: SceneRegistry, inst):
    light = inst.object.data
    if light.cycles.is_portal:
        registry.warn("Light portals are not supported")
        return []

    converted_matrix = convert_world_matrix(inst.matrix_world)

    if light.type in {"POINT", "SPOT"}:
        return [_build_point_light(registry, light, converted_matrix)]

    if light.type == "AREA":
        if not getattr(registry.settings, "enable_area_lights", True):
            return [_build_point_light(registry, light, converted_matrix)]
        scale_x, scale_y = _area_light_extents(registry, light)
        rect_node = _build_rectangle_light(registry, light, inst.object.name, converted_matrix, scale_x=scale_x, scale_y=scale_y)
        return [rect_node] if rect_node is not None else []

    if light.type == "SUN":
        if not getattr(registry.settings, "enable_area_lights", True):
            return [_build_point_light(registry, light, converted_matrix)]
        size = _sun_extents(light)
        rect_node = _build_rectangle_light(registry, light, inst.object.name, converted_matrix, scale_x=size, scale_y=size)
        return [rect_node] if rect_node is not None else []

    registry.warn(f"Light type {light.type} unsupported")
    return []


def export_lights(registry: SceneRegistry):
    result: list[XMLNode] = []

    if not getattr(registry.settings, "export_lights", True):
        return result

    for inst in registry.depsgraph.object_instances:
        object_eval = inst.object
        if object_eval is None or object_eval.type != 'LIGHT':
            continue
        if registry.settings.use_selection and not object_eval.original.select_get():
            continue
        if not registry.settings.use_selection and not inst.show_self:
            continue

        result.extend(export_light(registry, inst))

    return result
