import pytest
from openremote_client import AssetDatapoint
from pandas import DataFrame

from service_ml_forecast.ml.data_processing import (
    align_forecast_data,
    align_training_data,
    convert_datapoints_to_dataframe,
    resample_and_interpolate,
)
from service_ml_forecast.models.feature_data_wrappers import (
    AssetFeatureDatapoints,
    ForecastDataSet,
    TrainingDataSet,
)


def test_convert_empty_datapoints() -> None:
    """Empty input should return an empty DataFrame with correct columns."""
    df = convert_datapoints_to_dataframe([], time_col_name="timestamp", value_col_name="value")
    assert df.empty
    assert list(df.columns) == ["timestamp", "value"]


def test_resample_empty_datapoints() -> None:
    """Empty input should return None."""
    result = resample_and_interpolate([], frequency="1h", time_col_name="timestamp", value_col_name="value")
    assert result is None


def test_resample_single_unique_timestamp() -> None:
    """Multiple points with identical timestamps should aggregate to a single row."""
    datapoints = [
        AssetDatapoint(x=1_741_193_868_000, y=1.0),
        AssetDatapoint(x=1_741_193_868_000, y=2.0),
        AssetDatapoint(x=1_741_193_868_000, y=3.0),
    ]
    result = resample_and_interpolate(datapoints, frequency="1h", time_col_name="timestamp", value_col_name="value")
    assert result is not None
    assert len(result) == 1
    expected_mean = 2.0  # mean of 1, 2, 3
    assert result["value"].iloc[0] == expected_mean


def test_resample_unsorted_data() -> None:
    """Unsorted data should be sorted before resampling."""
    datapoints = [
        AssetDatapoint(x=1_741_193_868_000, y=3.0),
        AssetDatapoint(x=1_741_193_468_000, y=1.0),
        AssetDatapoint(x=1_741_193_668_000, y=2.0),
    ]
    result = resample_and_interpolate(datapoints, frequency="1h", time_col_name="timestamp", value_col_name="value")
    assert result is not None
    # After sorting and resampling, the values should be in chronological order
    timestamps = result["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_align_training_with_empty_regressor() -> None:
    """Empty regressor data should be excluded from training dataframe."""
    target = AssetFeatureDatapoints(
        feature_name="power",
        datapoints=[
            AssetDatapoint(x=1_741_193_868_000, y=10.0),
            AssetDatapoint(x=1_741_197_468_000, y=20.0),
        ],
    )
    empty_regressor = AssetFeatureDatapoints(feature_name="wind", datapoints=[])

    dataset = TrainingDataSet(target=target, regressors=[empty_regressor])
    result = align_training_data(dataset, frequency="1h")

    assert isinstance(result, DataFrame)
    assert "wind" not in result.columns


def test_align_training_with_regressor_outside_target_range() -> None:
    """Regressor that does not overlap with target range should be handled."""
    target = AssetFeatureDatapoints(
        feature_name="power",
        datapoints=[
            AssetDatapoint(x=1_741_193_868_000, y=10.0),
            AssetDatapoint(x=1_741_197_468_000, y=20.0),
        ],
    )
    # Regressor is far in the future, no overlap
    future_regressor = AssetFeatureDatapoints(
        feature_name="wind",
        datapoints=[
            AssetDatapoint(x=1_751_193_868_000, y=5.0),
            AssetDatapoint(x=1_751_197_468_000, y=6.0),
        ],
    )

    dataset = TrainingDataSet(target=target, regressors=[future_regressor])
    # This should raise because common date range will be empty / invalid
    with pytest.raises(ValueError):
        align_training_data(dataset, frequency="1h")


def test_align_forecast_with_empty_regressor() -> None:
    """Empty regressor in forecast dataset should be skipped gracefully."""
    future_df = DataFrame({"timestamp": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"]})
    future_df["timestamp"] = future_df["timestamp"].astype("datetime64[ns]")

    empty_regressor = AssetFeatureDatapoints(feature_name="wind", datapoints=[])
    forecast_dataset = ForecastDataSet(regressors=[empty_regressor])

    result = align_forecast_data(future_df, forecast_dataset, frequency="1h", config_id=__import__("uuid").uuid4())
    assert "wind" not in result.columns


def test_align_forecast_with_none_dataset() -> None:
    """None forecast dataset should return the original future dataframe."""
    future_df = DataFrame({"timestamp": ["2025-01-01 00:00:00"]})
    future_df["timestamp"] = future_df["timestamp"].astype("datetime64[ns]")

    result = align_forecast_data(future_df, None, frequency="1h", config_id=__import__("uuid").uuid4())
    assert result.equals(future_df)
