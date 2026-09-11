"""CLI tests: the demo exercises production APIs and shows mixed verdicts."""

from relate.cli import main


def test_demo_shows_mixed_verdicts(capsys):
    assert main(["demo"]) == 0
    output = capsys.readouterr().out
    assert "SOURCE" in output and "TARGET" in output
    assert "retrieval" in output and "PASS" in output
    assert "threshold_transfer" in output and "FAIL" in output
    assert "native target hash never reused" in output
    assert "COMPRESSION" in output
    assert "compression/retrieval" in output and "PASS" in output
    assert "OPERATOR" in output
    assert "operator/constant_delta" in output and "PASS" in output
    assert "operator/identity_map" in output and "FAIL" in output
    assert "lineage composes, permission does not" in output


def test_spaces_help(capsys):
    assert main(["spaces"]) == 0
    assert "space_hash" in capsys.readouterr().out
