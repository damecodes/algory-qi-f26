"""Minimal stand-in for the QI starter repo's _check helper.

Provides the Checker API the check_hwNN.py scripts use: load, exists, check,
assert_true, report. Replace with the official _check.py if you get it.
"""
import importlib.util
import pathlib
import traceback


class Checker:
    def __init__(self, title, target):
        self.title = title
        self.target = target
        self.passed = 0
        self.failed = 0
        self.module = None
        print(f"\n{title}\n{'=' * len(title)}")

    def load(self, checker_file):
        path = pathlib.Path(checker_file).parent / self.target
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        return module

    def _pass(self, name):
        self.passed += 1
        print(f"  PASS  {name}")

    def _fail(self, name, detail, note=None):
        self.failed += 1
        print(f"  FAIL  {name}")
        print(f"        {detail}")
        if note:
            print(f"        hint: {note}")

    def exists(self, name, attr):
        if hasattr(self.module, attr):
            self._pass(name)
        else:
            self._fail(name, f"{self.target} has no `{attr}`")

    def check(self, name, fn, expected, tol=1e-9, note=None):
        try:
            got = fn()
        except Exception as e:
            self._fail(name, f"raised {type(e).__name__}: {e}", note)
            return
        try:
            ok = abs(float(got) - float(expected)) <= tol
        except (TypeError, ValueError):
            ok = False
        if ok:
            self._pass(name)
        else:
            self._fail(name, f"got {got!r}, expected {expected!r} (tol {tol})", note)

    def assert_true(self, name, fn, note=None):
        try:
            ok = bool(fn())
        except Exception as e:
            self._fail(name, f"raised {type(e).__name__}: {e}\n"
                             + traceback.format_exc(limit=2), note)
            return
        if ok:
            self._pass(name)
        else:
            self._fail(name, "condition was False", note)

    def report(self):
        total = self.passed + self.failed
        print(f"\n{self.passed}/{total} checks passed.")
        return 0 if self.failed == 0 else 1
