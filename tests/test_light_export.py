import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _install_fake_bpy():
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.types = types.SimpleNamespace(
        Light=type("Light", (), {}),
        Object=type("Object", (), {}),
    )
    fake_bpy.ops = types.SimpleNamespace()
    fake_bpy.context = types.SimpleNamespace()
    fake_bpy.props = types.ModuleType("bpy.props")
    fake_bpy.props.StringProperty = lambda **kwargs: None
    fake_bpy.utils = types.SimpleNamespace(register_class=lambda cls: None, unregister_class=lambda cls: None)
    return {
        "bpy": fake_bpy,
        "bpy.types": fake_bpy.types,
        "bpy.props": fake_bpy.props,
    }


class _Vector:
    def __init__(self, seq):
        self._data = list(seq)

    def __mul__(self, scalar):
        return _Vector([component * scalar for component in self._data])

    __rmul__ = __mul__

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    @property
    def z(self):
        return self._data[2]

    @property
    def length(self):
        return sum(v * v for v in self._data) ** 0.5

    @property
    def length_squared(self):
        return sum(v * v for v in self._data)


class _Matrix:
    def __init__(self, rows):
        self._rows = [list(row) for row in rows]

    def __matmul__(self, other):
        result = []
        for r in range(len(self._rows)):
            new_row = []
            for c in range(len(other._rows[0])):
                new_row.append(sum(self._rows[r][k] * other._rows[k][c] for k in range(len(other._rows))))
            result.append(new_row)
        return _Matrix(result)

    def to_3x3(self):
        return _Matrix([row[:3] for row in self._rows[:3]])

    @property
    def translation(self):
        return _Vector([self._rows[0][3], self._rows[1][3], self._rows[2][3]])

    @property
    def col(self):
        cols = []
        for c in range(len(self._rows[0])):
            cols.append(_Vector([self._rows[r][c] for r in range(min(3, len(self._rows)))]))
        return cols


def _install_fake_mathutils():
    mathutils = types.ModuleType("mathutils")
    mathutils.Matrix = _Matrix
    mathutils.Vector = _Vector
    mathutils.Quaternion = type("Quaternion", (), {})  # placeholder to satisfy imports
    return {"mathutils": mathutils}


def _load_lightwave_module(name, path, search_path):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=search_path
    )
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


class _DummyDepsgraph:
    def __init__(self, *instances):
        self.object_instances = list(instances)
        self.scene = types.SimpleNamespace(frame_current=0)


class _DummyOperator:
    def __init__(self):
        self.messages = []

    def report(self, levels, message):
        self.messages.append((levels, message))


class _DummySettings:
    enable_area_lights = True
    export_lights = True
    use_selection = False


class LightExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.update(_install_fake_mathutils())
        sys.modules.update(_install_fake_bpy())

        repo_root = Path(__file__).resolve().parent.parent / "Nomad Path Tracer"
        package_root = repo_root / "lightwave_blender"
        base_package = types.ModuleType("lightwave_blender")
        base_package.__path__ = [str(package_root)]
        sys.modules["lightwave_blender"] = base_package

        cls.utils_module = _load_lightwave_module("lightwave_blender.utils", package_root / "utils.py", base_package.__path__)
        cls.registry_module = _load_lightwave_module("lightwave_blender.registry", package_root / "registry.py", base_package.__path__)
        cls.light_module = _load_lightwave_module("lightwave_blender.light", package_root / "light.py", base_package.__path__)

    def _build_light_instance(self, light_type, matrix_rows, **kwargs):
        import types

        light = types.SimpleNamespace(
            type=light_type,
            shadow_soft_size=kwargs.get("shadow_soft_size", 0.1),
            energy=kwargs.get("energy", 10.0),
            color=kwargs.get("color", (1.0, 1.0, 1.0)),
            shape=kwargs.get("shape", "SQUARE"),
            size=kwargs.get("size", 2.0),
            size_y=kwargs.get("size_y", 2.0),
            cycles=types.SimpleNamespace(is_portal=False),
            angle=kwargs.get("angle", 0.0),
        )
        obj = types.SimpleNamespace(data=light, type="LIGHT", name="TestLight", original=types.SimpleNamespace(select_get=lambda: True))
        inst = types.SimpleNamespace(object=obj, matrix_world=self.utils_module.mathutils.Matrix(matrix_rows), show_self=True)
        return inst

    def test_point_light_uses_converted_matrix(self):
        inst = self._build_light_instance("POINT", (
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 2.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ))
        registry = self.registry_module.SceneRegistry(
            path="/tmp",
            depsgraph=_DummyDepsgraph(inst),
            settings=_DummySettings(),
            op=_DummyOperator(),
        )

        lights = self.light_module.export_light(registry, inst)

        self.assertEqual(len(lights), 1)
        position = lights[0].attributes["position"]
        self.assertEqual(position, "1,3,-2")

    def test_area_light_orientation_matches_converted_space(self):
        inst = self._build_light_instance("AREA", (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 1.5),
            (0.0, 0.0, 1.0, 2.5),
            (0.0, 0.0, 0.0, 1.0),
        ))
        registry = self.registry_module.SceneRegistry(
            path="/tmp",
            depsgraph=_DummyDepsgraph(inst),
            settings=_DummySettings(),
            op=_DummyOperator(),
        )

        lights = self.light_module.export_light(registry, inst)

        self.assertEqual(len(lights), 1)
        rect = lights[0]
        self.assertEqual(rect.attributes["position"], "0,2.5,-1.5")
        self.assertEqual(rect.attributes["u"], "1,0,0")
        self.assertEqual(rect.attributes["v"], "0,0,-1")


if __name__ == "__main__":
    unittest.main()
