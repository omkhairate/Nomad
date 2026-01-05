import os
import sys
import tempfile
import types
import unittest
import importlib.util
from pathlib import Path


def _install_fake_bpy():
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.types = types.SimpleNamespace(
        Image=type("Image", (), {}),
        AddonPreferences=type("AddonPreferences", (), {}),
        Object=type("Object", (), {}),
        Node=type("Node", (), {}),
        NodeTree=type("NodeTree", (), {}),
    )
    props_module = types.ModuleType("bpy.props")
    props_module.StringProperty = lambda **kwargs: None
    fake_bpy.props = props_module
    fake_bpy.path = types.SimpleNamespace(
        abspath=lambda path, library=None: path,
        resolve_ncase=lambda path: path,
        relpath=lambda path, start=None: path,
        basename=lambda path: os.path.basename(path),
    )
    fake_bpy.utils = types.SimpleNamespace(register_class=lambda cls: None, unregister_class=lambda cls: None)
    fake_bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons={"lightwave_blender": types.SimpleNamespace(preferences=types.SimpleNamespace(tex_dir_name="textures", mesh_dir_name="meshes"))})
    )
    sys.modules["bpy"] = fake_bpy
    sys.modules["bpy.props"] = props_module
    sys.modules["bpy.types"] = fake_bpy.types


def _install_fake_mathutils():
    mathutils = types.ModuleType("mathutils")

    class _Quaternion:
        def __init__(self, *args, **kwargs):
            pass

        def __matmul__(self, other):
            return self

    class _Vector:
        @staticmethod
        def Fill(length, value):
            return [value] * length

    class _Matrix:
        @staticmethod
        def LocRotScale(loc, rot, sca):
            return (loc, rot, sca)

    mathutils.Quaternion = _Quaternion
    mathutils.Vector = _Vector
    mathutils.Matrix = _Matrix
    sys.modules["mathutils"] = mathutils


class _PixelBuffer:
    def __init__(self, data=None):
        self._data = data or []

    def __len__(self):
        return len(self._data)

    def foreach_get(self, out):
        for idx in range(min(len(out), len(self._data))):
            out[idx] = self._data[idx]


class _FakeImage:
    def __init__(self):
        self.name = "EmptyImage"
        self.has_data = False
        self.pixels = _PixelBuffer([])
        self.size = (1024, 1024)
        self.filepath_raw = ""
        self.file_format = "PNG"
        self._saved = False
        self._render_saved = False

    def reload(self):
        # Keep pixels empty to simulate an unloaded image that cannot be reloaded
        return None

    def update(self):
        # Keep pixels empty to simulate an unloaded image that cannot be updated
        return None

    def save(self):
        self._saved = True

    def save_render(self, path):
        self._render_saved = True


class _DummyDepsgraph:
    scene = None


class _DummySettings:
    overwrite_existing_textures = False


class _DummyOperator:
    def __init__(self):
        self.messages = []

    def report(self, levels, message):
        self.messages.append((levels, message))


class ExportImageTests(unittest.TestCase):
    def setUp(self):
        _install_fake_bpy()
        _install_fake_mathutils()
        repo_root = Path(__file__).resolve().parent.parent
        package_root = repo_root / "Nomad Path Tracer" / "lightwave_blender"
        base_package = types.ModuleType("lightwave_blender")
        base_package.__path__ = [str(package_root)]
        sys.modules["lightwave_blender"] = base_package

        def _load_module(name, path):
            spec = importlib.util.spec_from_file_location(
                name, path, submodule_search_locations=base_package.__path__
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            return module

        registry_module = _load_module("lightwave_blender.registry", package_root / "registry.py")
        self.node_module = _load_module("lightwave_blender.node", package_root / "node.py")
        self.registry_cls = registry_module.SceneRegistry

    def test_unloaded_image_is_skipped_with_error(self):
        dummy_op = _DummyOperator()
        registry = self.registry_cls(
            path=tempfile.gettempdir(),
            depsgraph=_DummyDepsgraph(),
            settings=_DummySettings(),
            op=dummy_op,
        )
        image = _FakeImage()
        export_path = os.path.join(tempfile.gettempdir(), "should_not_write.png")

        result = self.node_module._export_image(registry, image, export_path, is_f32=False, keep_format=False)

        self.assertFalse(result)
        self.assertTrue(any("has no pixel data loaded" in msg for _, msg in dummy_op.messages))
        self.assertFalse(image._saved)
        self.assertFalse(image._render_saved)


if __name__ == "__main__":
    unittest.main()
