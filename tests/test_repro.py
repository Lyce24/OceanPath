"""
Tests for oceanpath.utils.repro — reproducibility and provenance.

Covers:
  - Config fingerprinting (deterministic, sensitive to changes)
  - Manifest hashing (order-independent, detects data changes)
  - Environment verification against uv.lock
  - Provenance capture
"""

import json

import pytest


class TestConfigFingerprint:
    def test_deterministic(self):
        """Same config → same fingerprint."""
        from oceanpath.utils.repro import config_fingerprint

        cfg = {"model": {"arch": "abmil"}, "training": {"lr": 1e-4}}
        fp1 = config_fingerprint(cfg)
        fp2 = config_fingerprint(cfg)
        assert fp1 == fp2

    def test_different_config_different_fingerprint(self):
        """Changing any value changes the fingerprint."""
        from oceanpath.utils.repro import config_fingerprint

        cfg1 = {"model": {"arch": "abmil"}, "training": {"lr": 1e-4}}
        cfg2 = {"model": {"arch": "abmil"}, "training": {"lr": 2e-4}}
        assert config_fingerprint(cfg1) != config_fingerprint(cfg2)

    def test_key_order_irrelevant(self):
        """Dict key order doesn't affect fingerprint (sorted internally)."""
        from oceanpath.utils.repro import config_fingerprint

        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert config_fingerprint(cfg1) == config_fingerprint(cfg2)

    def test_returns_12_char_hex(self):
        from oceanpath.utils.repro import config_fingerprint

        fp = config_fingerprint({"x": 1})
        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)


class TestManifestHash:
    def test_same_file_same_hash(self, tmp_path):
        from oceanpath.utils.repro import manifest_hash

        csv = tmp_path / "manifest.csv"
        csv.write_text("filename,label\nslide_a.svs,0\nslide_b.svs,1\n")

        h1 = manifest_hash(csv)
        h2 = manifest_hash(csv)
        assert h1 == h2

    def test_row_order_independent(self, tmp_path):
        """Same rows in different order → same hash."""
        from oceanpath.utils.repro import manifest_hash

        csv1 = tmp_path / "v1.csv"
        csv1.write_text("filename,label\nslide_a.svs,0\nslide_b.svs,1\n")

        csv2 = tmp_path / "v2.csv"
        csv2.write_text("filename,label\nslide_b.svs,1\nslide_a.svs,0\n")

        assert manifest_hash(csv1) == manifest_hash(csv2)

    def test_different_data_different_hash(self, tmp_path):
        from oceanpath.utils.repro import manifest_hash

        csv1 = tmp_path / "v1.csv"
        csv1.write_text("filename,label\nslide_a.svs,0\n")

        csv2 = tmp_path / "v2.csv"
        csv2.write_text("filename,label\nslide_a.svs,0\nslide_c.svs,2\n")

        assert manifest_hash(csv1) != manifest_hash(csv2)

    def test_missing_file(self, tmp_path):
        from oceanpath.utils.repro import manifest_hash

        result = manifest_hash(tmp_path / "nonexistent.csv")
        assert result == "missing"

    def test_empty_file(self, tmp_path):
        from oceanpath.utils.repro import manifest_hash

        csv = tmp_path / "empty.csv"
        csv.write_text("filename,label\n")
        assert manifest_hash(csv) == "empty"


class TestVerifyEnvironment:
    def test_missing_lockfile(self, tmp_path):
        from oceanpath.utils.repro import verify_environment

        result = verify_environment(lockfile=tmp_path / "missing.lock")
        assert result["status"] == "missing"

    def test_valid_lockfile(self, tmp_path):
        """Verify returns ok or mismatch (not crash) with a real-ish lockfile."""
        from oceanpath.utils.repro import verify_environment

        lock = tmp_path / "uv.lock"
        lock.write_text(
            "version = 1\n\n"
            '[[package]]\nname = "numpy"\nversion = "99.99.99"\n\n'
            '[[package]]\nname = "torch"\nversion = "99.99.99"\n'
        )

        result = verify_environment(lockfile=lock)
        # These fake versions won't match, so expect mismatch
        assert result["status"] in ("ok", "mismatch")
        assert "mismatches" in result

    def test_strict_mode_raises(self, tmp_path):
        from oceanpath.utils.repro import verify_environment

        lock = tmp_path / "uv.lock"
        lock.write_text('[[package]]\nname = "fake_package_xyz"\nversion = "0.0.0"\n')

        with pytest.raises(RuntimeError, match="diverges"):
            verify_environment(lockfile=lock, strict=True)

    def test_parse_uv_lock(self, tmp_path):
        from oceanpath.utils.repro import _parse_uv_lock

        lock = tmp_path / "uv.lock"
        lock.write_text(
            "version = 1\n\n"
            '[[package]]\nname = "numpy"\nversion = "2.2.6"\n\n'
            '[[package]]\nname = "torch"\nversion = "2.10.0"\n\n'
            '[[package]]\nname = "scikit-learn"\nversion = "1.7.2"\n'
        )

        packages = _parse_uv_lock(lock)
        assert packages["numpy"] == "2.2.6"
        assert packages["torch"] == "2.10.0"
        assert packages["scikit_learn"] == "1.7.2"  # dash → underscore


class TestCaptureProvenance:
    def test_returns_dict_with_required_keys(self):
        from oceanpath.utils.repro import capture_provenance

        prov = capture_provenance()
        assert "timestamp" in prov
        assert "python_version" in prov
        assert "torch_version" in prov
        assert "git" in prov

    def test_with_config(self):
        from oceanpath.utils.repro import capture_provenance

        cfg = {"model": {"arch": "abmil"}}
        prov = capture_provenance(cfg=cfg)
        assert "config_fingerprint" in prov
        assert len(prov["config_fingerprint"]) == 12

    def test_with_csv_path(self, tmp_path):
        from oceanpath.utils.repro import capture_provenance

        csv = tmp_path / "manifest.csv"
        csv.write_text("filename,label\nslide.svs,0\n")

        prov = capture_provenance(csv_path=csv)
        assert "manifest_hash" in prov
        assert prov["manifest_hash"] != "missing"

    def test_save_provenance(self, tmp_path):
        from oceanpath.utils.repro import capture_provenance, save_provenance

        prov = capture_provenance()
        out = tmp_path / "sub" / "provenance.json"
        save_provenance(prov, out)

        assert out.is_file()
        loaded = json.loads(out.read_text())
        assert "timestamp" in loaded
