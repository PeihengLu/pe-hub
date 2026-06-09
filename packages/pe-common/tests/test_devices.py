"""Tests for device discovery helpers."""
from __future__ import annotations

import pytest

from pe_common.devices import (
    AUTO_DEVICE,
    default_device_id,
    format_devices_for_cli,
    list_device_ids,
    list_devices,
    resolve_device_id,
)


def test_list_devices_includes_cpu():
    devices = list_devices()
    ids = [device.device_id for device in devices]
    assert "cpu" in ids


def test_default_device_is_known():
    device_id = default_device_id()
    assert device_id in list_device_ids(include_cpu=True)


def test_resolve_auto_device():
    assert resolve_device_id(AUTO_DEVICE) == default_device_id()
    assert resolve_device_id(None) == default_device_id()


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device_id("not-a-real-device")


def test_format_devices_for_cli():
    text = format_devices_for_cli()
    assert "Available devices:" in text
    assert "Default:" in text
