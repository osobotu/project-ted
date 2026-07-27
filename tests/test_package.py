from importlib.metadata import version

import project_ted


def test_package_is_installable() -> None:
    assert project_ted.__doc__
    assert version("project-ted") == "0.1.0"
