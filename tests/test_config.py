"""Tests for configuration system."""

from pathlib import Path

from energy_forecasting.config import PROJECT_ROOT, Settings, load_settings, settings


class TestSettings:
    """Verify that config loads and validates correctly."""

    def test_settings_load(self):
        """Settings object should load without errors."""
        assert settings is not None

    def test_project_root_exists(self):
        """PROJECT_ROOT should point to actual directory."""
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "configs" / "default.yaml").exists()

    def test_entso_e_config(self):
        """ENTSO-E section should have valid values."""
        assert settings.entso_e.area_code == "10YGR-HTSO-----Y"
        assert settings.entso_e.base_url.startswith("https://")

    def test_weather_locations(self):
        """Weather config should have at least one location."""
        assert len(settings.weather.locations) > 0

    def test_weather_weights_sum_to_one(self):
        """Location weights should sum to 1.0."""
        total = sum(loc.weight for loc in settings.weather.locations)
        assert abs(total - 1.0) < 0.01

    def test_data_paths_are_relative(self):
        """Data paths should be relative, not absolute."""
        assert not settings.data.raw_dir.is_absolute()