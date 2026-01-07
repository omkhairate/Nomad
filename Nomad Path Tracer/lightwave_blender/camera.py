import mathutils
from .utils import *
from .registry import SceneRegistry
from .xml_node import XMLNode


def _get_camera_frames(camera, scene):
    frames = set()

    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for keyframe_point in fcurve.keyframe_points:
                frames.add(int(keyframe_point.co[0]))

    if len(frames) == 0:
        frames.add(scene.frame_current)

    return sorted(frames)


def export_camera(registry: SceneRegistry):
    camera = registry.scene.camera
    if camera is None:
        registry.error("Your scene needs a camera!")
        return []

    camera_path = XMLNode("CameraPath")

    current_frame = registry.scene.frame_current
    for frame in _get_camera_frames(camera, registry.scene):
        registry.scene.frame_set(frame)

        cam_eval = camera.evaluated_get(registry.depsgraph)
        matrix = orient_camera(cam_eval.matrix_world, skip_scale=True)

        position = matrix.to_translation()
        rotation = matrix.to_quaternion()
        forward = rotation @ mathutils.Vector((0.0, 0.0, -1.0))
        look_at = position + forward
        up = rotation @ mathutils.Vector((0.0, 1.0, 0.0))
        up.normalize()

        camera_path.add("Keyframe",
                        frame=frame,
                        position=str_flat_array(position),
                        lookAt=str_flat_array(look_at),
                        up=str_flat_array(up))

    registry.scene.frame_set(current_frame)

    return [camera_path]
