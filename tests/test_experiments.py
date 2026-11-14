import csv

import pytest

from gpumodel.ncu import METRICS, extract_csv, read_raw_counters
from gpumodel.report import group_runs, measured_bytes


def raw_csv(path, rows, unit="byte"):
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows([["ID", "Kernel Name", *METRICS],
                                     ["", "", unit, unit], *rows])
    return path


def test_complete_dram_export(tmp_path):
    raw = raw_csv(tmp_path / "raw.csv", [[0, "forward", "1,024", 0],
                                        [1, "backward", 12, 340],
                                        [2, "update", 5, 80],
                                        ["Summary", "", 1041, 420]])
    output = tmp_path / "measured.csv"
    extract_csv(raw, output, "desktop", "mlp", gpu_id="GPU-example")
    group = group_runs([run()])
    assert measured_bytes(output, group) == {("desktop", "mlp", "device"): 1461}
    with pytest.raises(ValueError, match="every plotted"):
        measured_bytes(output, group_runs([run(), dict(run(), model="cnn")]))
    with pytest.raises(ValueError, match="batch_size"):
        measured_bytes(output, group_runs([run(batch_size=8)]))
    raw.unlink()
    with pytest.raises(FileNotFoundError):
        measured_bytes(output, group)


@pytest.mark.parametrize("rows,unit", [
    ([], "byte"),
    ([[0, "a", 20, 10], [1, "b", "n/a", 40]], "byte"),
    ([[0, "a", 20, 10], [1, "b", 20]], "byte"),
    ([[0, "a", 20, 10], [2, "b", 20, 30]], "byte"),
    ([[0, "a", 20, 10], [0, "b", 20, 30]], "byte"),
    ([[0, "a", "Infinity", 10]], "byte"),
    ([[0, "a", 20, 0]], "byte"),
    ([[0, "a", 20, 10]], "Kbyte"),
])
def test_reject_incomplete_or_invalid_dram(tmp_path, rows, unit):
    with pytest.raises(ValueError):
        read_raw_counters(raw_csv(tmp_path / "invalid.csv", rows, unit))


def run(**overrides):
    return dict(
        environment="desktop", model="mlp", data_mode="device", batch_size=32,
        seed=1337, precision="FP32 (TF32 disabled)", warmup_steps=100,
        measured_steps=10000, image_shape=[3, 64, 64],
        hardware={"gpu_uuid": "GPU-example", "git_commit": "example"},
    ) | overrides


@pytest.mark.parametrize("change", [{"batch_size": 8}, {"measured_steps": 5}, {"seed": 1}])
def test_mixed_configuration_rejected(change):
    with pytest.raises(ValueError, match="Mixed run configurations"):
        group_runs([run(), run(environment="other", **change)])
