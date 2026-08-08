"""Lenient LeRobot OpenCV camera support.

The Intel D435i UVC color node rejects manual width/height/fps settings, so
LeRobot's stock ``opencv`` camera aborts in ``_configure_capture_settings``.
This module monkey-patches that method: if the device refuses the settings,
we log a warning and continue with the device defaults (the D435i color node
delivers 640x480@30 out of the box). No extra dependency is required.

Importing this module applies the patch; the plugin's __init__.py does that.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_MARK = "_uf_lenient_opencv_patch"


def _install_lenient_opencv_camera() -> None:
    try:
        from lerobot.cameras.opencv import camera_opencv as cam_mod
    except Exception:
        logger.debug("lerobot opencv camera module not found; patch skipped")
        return

    cls = getattr(cam_mod, "OpenCVCamera", None)
    if cls is None:
        logger.debug("OpenCVCamera not found; patch skipped")
        return

    original = getattr(cls, "_configure_capture_settings", None)
    if original is None or getattr(original, _PATCH_MARK, False):
        return

    def _lenient_configure(self) -> None:
        try:
            original(self)
        except Exception as exc:  # noqa: BLE001 - keep recording alive
            logger.warning(
                "OpenCVCamera(%s) could not apply all capture settings (%s); "
                "continuing with device defaults",
                getattr(self, "index_or_path", "?"),
                exc,
            )

    setattr(_lenient_configure, _PATCH_MARK, True)
    cls._configure_capture_settings = _lenient_configure
    logger.debug("Applied lenient OpenCV capture-settings patch")


_install_lenient_opencv_camera()

__all__: list[str] = []
