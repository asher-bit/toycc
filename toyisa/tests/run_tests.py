"""toyisa 零依赖测试运行器。

机器上没有 pytest 也能跑完整个回归套件(课程要求"只需要 numpy")。
用法:
    python toyisa/tests/run_tests.py
装了 pytest 也可以直接: python -m pytest toyisa/tests
"""
import importlib.util
import sys
import traceback
import types


class _Raises:
    """pytest.raises 的极简替身。"""

    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(f"期望抛出 {self.exc.__name__}, 但没有抛出")
        self.value = exc_val
        return issubclass(exc_type, self.exc)


def main():
    sys.path.insert(0, ".")          # 保证从项目根目录运行时能 import toyisa
    # pytest 已装则直接用; 否则注入替身模块
    try:
        import pytest  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pytest")
        stub.raises = _Raises
        sys.modules["pytest"] = stub

    spec = importlib.util.spec_from_file_location(
        "test_toyisa", "toyisa/tests/test_toyisa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tests = sorted(n for n in dir(mod) if n.startswith("test_"))
    passed, failed = [], []
    for name in tests:
        try:
            getattr(mod, name)()
            passed.append(name)
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc(limit=3)
    print(f"\n{len(passed)} passed, {len(failed)} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
