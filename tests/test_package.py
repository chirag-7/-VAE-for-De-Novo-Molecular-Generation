"""Smoke tests for the molgen package itself."""

import molgen


def test_version_is_nonempty_string():
    assert isinstance(molgen.__version__, str)
    assert molgen.__version__


def test_submodules_import():
    # Importing must be free of side effects (no dataset generation, etc.).
    import molgen.generate  # noqa: F401
    import molgen.interpolate  # noqa: F401
    import molgen.synthetic  # noqa: F401
    import molgen.vae  # noqa: F401
