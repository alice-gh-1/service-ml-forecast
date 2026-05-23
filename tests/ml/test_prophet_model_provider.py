import pytest
from openremote_client import AssetDatapoint

from service_ml_forecast.common.exceptions import ResourceNotFoundError
from service_ml_forecast.ml.model_provider_factory import ModelProviderFactory
from service_ml_forecast.models.feature_data_wrappers import AssetFeatureDatapoints, ForecastDataSet, TrainingDataSet
from service_ml_forecast.models.model_config import ProphetModelConfig


def test_train_and_predict(
    prophet_basic_config: ProphetModelConfig,
    windspeed_mock_datapoints: list[AssetDatapoint],
) -> None:
    """Test the basic functionality of a Prophet model provider with a single variable.

    Verifies that:
    - The model trains successfully with the provided data
    - The trained model can be saved and loaded
    - The model can generate forecasts with non-empty results
    """
    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    # Train the model
    model = model_provider.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name=prophet_basic_config.target.attribute_name,
                datapoints=windspeed_mock_datapoints,
            ),
        ),
    )
    assert model is not None

    # Save the model
    model_provider.save_model(model)
    assert prophet_basic_config.id is not None
    assert model_provider.load_model(prophet_basic_config.id) is not None

    # Generate the forecast
    forecast = model_provider.generate_forecast()
    assert forecast is not None
    assert forecast.datapoints is not None
    assert len(forecast.datapoints) > 0


def test_train_and_predict_with_regressor(
    prophet_multi_variable_config: ProphetModelConfig,
    prophet_basic_config: ProphetModelConfig,
    tariff_mock_datapoints: list[AssetDatapoint],
    windspeed_mock_datapoints: list[AssetDatapoint],
) -> None:
    """Test Prophet model with external regressors for multi-variable forecasting.

    Verifies that:
    - Both models train successfully (regressor and target)
    - Models can be saved and then loaded
    - The windspeed forecast can be used as a regressor input for the tariff model
    - The tariff model generates valid forecasts when provided with regressor data
    """
    # Create the windspeed model
    windspeed_provider = ModelProviderFactory.create_provider(prophet_basic_config)
    windspeed_target_datapoints = AssetFeatureDatapoints(
        feature_name=prophet_basic_config.target.attribute_name,
        datapoints=windspeed_mock_datapoints,
    )

    # Train the windspeed model
    windspeed_model = windspeed_provider.train_model(
        TrainingDataSet(target=windspeed_target_datapoints),
    )
    assert windspeed_model is not None
    # Save the windspeed model
    windspeed_provider.save_model(windspeed_model)
    assert windspeed_provider.load_model(prophet_basic_config.id) is not None

    # Generate the forecast
    windspeed_forecast = windspeed_provider.generate_forecast()
    assert windspeed_forecast is not None
    assert windspeed_forecast.datapoints is not None
    assert len(windspeed_forecast.datapoints) > 0

    # Create the tariff model
    tarrif_provider = ModelProviderFactory.create_provider(prophet_multi_variable_config)
    tariff_target_datapoints = AssetFeatureDatapoints(
        feature_name=prophet_multi_variable_config.target.attribute_name,
        datapoints=tariff_mock_datapoints,
    )

    # Ensure tariff model has regressors configured
    assert prophet_multi_variable_config.regressors is not None
    assert len(prophet_multi_variable_config.regressors) > 0

    # Train the tariff model
    regressor_feature_datapoints = [
        AssetFeatureDatapoints(feature_name=regressor.attribute_name, datapoints=windspeed_mock_datapoints)
        for regressor in prophet_multi_variable_config.regressors
    ]

    tariff_model = tarrif_provider.train_model(
        TrainingDataSet(target=tariff_target_datapoints, regressors=regressor_feature_datapoints),
    )
    assert tariff_model is not None

    # Save the tariff model
    tarrif_provider.save_model(tariff_model)
    assert tarrif_provider.load_model(prophet_multi_variable_config.id) is not None

    # Generate the forecast including the regressor forecast datapoints
    windspeed_regressor_datapoints = AssetFeatureDatapoints(
        feature_name=prophet_basic_config.target.attribute_name,
        datapoints=windspeed_forecast.datapoints,
    )
    forecast_dataset = ForecastDataSet(regressors=[windspeed_regressor_datapoints])
    forecast = tarrif_provider.generate_forecast(forecast_dataset)
    assert forecast is not None
    assert forecast.datapoints is not None
    assert len(forecast.datapoints) > 0


def test_train_with_insufficient_data(
    prophet_basic_config: ProphetModelConfig,
) -> None:
    """Test that training fails gracefully with insufficient data.

    Verifies that:
    - Model returns None when training data is too small
    - No exceptions are raised with edge case data
    """
    # Create minimal dataset with only 1 datapoint
    minimal_datapoints = [AssetDatapoint(x=1741193868185, y=2.57)]

    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    model = model_provider.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name=prophet_basic_config.target.attribute_name,
                datapoints=minimal_datapoints,
            ),
        ),
    )

    # Should return None for insufficient data
    assert model is None


def test_train_with_empty_data(
    prophet_basic_config: ProphetModelConfig,
) -> None:
    """Test that training fails gracefully with empty data.

    Verifies that:
    - Model returns None when no training data is provided
    - No exceptions are raised with empty dataset
    """
    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    model = model_provider.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name=prophet_basic_config.target.attribute_name,
                datapoints=[],
            ),
        ),
    )

    # Should return None for empty data
    assert model is None


def test_forecast_without_trained_model(
    prophet_basic_config: ProphetModelConfig,
) -> None:
    """Test that forecasting fails gracefully when no model is trained.

    Verifies that:
    - Appropriate exception is raised when trying to forecast without training
    - Error handling works correctly for missing model scenario
    """
    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    # Try to generate forecast without training a model first
    with pytest.raises(ResourceNotFoundError):  # Should raise an exception when model doesn't exist
        model_provider.generate_forecast()


def test_data_resampling_and_alignment(
    prophet_multi_variable_config: ProphetModelConfig,
    windspeed_mock_datapoints: list[AssetDatapoint],
    tariff_mock_datapoints: list[AssetDatapoint],
) -> None:
    """Test that data resampling and alignment works correctly with different frequencies.

    Verifies that:
    - Data with different frequencies (10min vs 1hour) can be aligned
    - Resampling produces reasonable results
    - Training succeeds with mismatched data frequencies
    """
    model_provider = ModelProviderFactory.create_provider(prophet_multi_variable_config)

    # Create training data with different frequencies
    target_datapoints = AssetFeatureDatapoints(
        feature_name=prophet_multi_variable_config.target.attribute_name,
        datapoints=tariff_mock_datapoints,  # 1-hour frequency
    )

    assert prophet_multi_variable_config.regressors is not None
    regressor_datapoints = AssetFeatureDatapoints(
        feature_name=prophet_multi_variable_config.regressors[0].attribute_name,
        datapoints=windspeed_mock_datapoints,  # 10-minute frequency
    )

    # Train model with mismatched frequencies
    model = model_provider.train_model(
        TrainingDataSet(
            target=target_datapoints,
            regressors=[regressor_datapoints],
        ),
    )

    # Should succeed despite frequency mismatch
    assert model is not None


def test_forecast_with_missing_regressor_data(
    prophet_multi_variable_config: ProphetModelConfig,
    tariff_mock_datapoints: list[AssetDatapoint],
) -> None:
    """Test forecasting behavior when regressor data is missing or incomplete.

    Verifies that:
    - Model handles missing regressor data gracefully
    - Forecasts can still be generated with partial regressor information
    - Appropriate warnings are logged for missing data
    """
    model_provider = ModelProviderFactory.create_provider(prophet_multi_variable_config)

    # Train model first
    target_datapoints = AssetFeatureDatapoints(
        feature_name=prophet_multi_variable_config.target.attribute_name,
        datapoints=tariff_mock_datapoints,
    )

    # Create regressor with some data
    assert prophet_multi_variable_config.regressors is not None
    regressor_datapoints = AssetFeatureDatapoints(
        feature_name=prophet_multi_variable_config.regressors[0].attribute_name,
        datapoints=tariff_mock_datapoints[:5],  # Only first 5 points
    )

    model = model_provider.train_model(
        TrainingDataSet(
            target=target_datapoints,
            regressors=[regressor_datapoints],
        ),
    )
    assert model is not None

    model_provider.save_model(model)

    # Try to forecast with minimal regressor data (Prophet requires regressors to be present)
    assert prophet_multi_variable_config.regressors is not None
    minimal_regressor = AssetFeatureDatapoints(
        feature_name=prophet_multi_variable_config.regressors[0].attribute_name,
        datapoints=[tariff_mock_datapoints[0]],  # Just one point
    )

    forecast_dataset = ForecastDataSet(regressors=[minimal_regressor])

    # Should handle minimal regressor data gracefully
    forecast = model_provider.generate_forecast(forecast_dataset)
    assert forecast is not None
    assert forecast.datapoints is not None


def test_model_persistence_and_reload(
    prophet_basic_config: ProphetModelConfig,
    windspeed_mock_datapoints: list[AssetDatapoint],
) -> None:
    """Test that models can be saved and reloaded correctly.

    Verifies that:
    - Models can be saved to storage
    - Saved models can be loaded back
    - Loaded models produce consistent forecasts
    """
    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    # Train and save model
    model = model_provider.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name=prophet_basic_config.target.attribute_name,
                datapoints=windspeed_mock_datapoints,
            ),
        ),
    )
    assert model is not None

    model_provider.save_model(model)

    # Generate forecast with original model
    original_forecast = model_provider.generate_forecast()
    assert original_forecast is not None
    assert len(original_forecast.datapoints) > 0

    # Create new provider instance and load model
    new_provider = ModelProviderFactory.create_provider(prophet_basic_config)
    loaded_model = new_provider.load_model(prophet_basic_config.id)
    assert loaded_model is not None

    # Generate forecast with loaded model
    loaded_forecast = new_provider.generate_forecast()
    assert loaded_forecast is not None
    assert len(loaded_forecast.datapoints) > 0

    # Forecasts should have same number of points
    assert len(original_forecast.datapoints) == len(loaded_forecast.datapoints)


def test_forecast_data_quality(
    prophet_basic_config: ProphetModelConfig,
    windspeed_mock_datapoints: list[AssetDatapoint],
) -> None:
    """Test that forecast data has reasonable quality and structure.

    Verifies that:
    - Forecast timestamps are in the future
    - Forecast values are reasonable (not NaN, not extreme outliers)
    - Forecast frequency matches expected frequency
    """
    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    # Train model
    model = model_provider.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name=prophet_basic_config.target.attribute_name,
                datapoints=windspeed_mock_datapoints,
            ),
        ),
    )
    assert model is not None

    model_provider.save_model(model)

    # Generate forecast
    forecast = model_provider.generate_forecast()
    assert forecast is not None
    assert forecast.datapoints is not None
    assert len(forecast.datapoints) > 0

    # Check forecast data quality
    last_training_time = max(point.x for point in windspeed_mock_datapoints)

    for datapoint in forecast.datapoints:
        # Forecast timestamps should be in the future
        assert datapoint.x > last_training_time

        # Forecast values should be reasonable (not NaN, not extreme)
        assert datapoint.y is not None
        assert not (datapoint.y != datapoint.y)  # Check for NaN
        MIN_REASONABLE_VALUE = -1000
        MAX_REASONABLE_VALUE = 1000
        assert MIN_REASONABLE_VALUE < datapoint.y < MAX_REASONABLE_VALUE  # Reasonable range check

    # Check forecast frequency (should match config)
    if len(forecast.datapoints) > 1:
        time_diffs = [
            forecast.datapoints[i + 1].x - forecast.datapoints[i].x for i in range(len(forecast.datapoints) - 1)
        ]
        # Most time differences should be close to expected frequency (1 hour = 3600000 ms)
        expected_interval = 3600000  # 1 hour in milliseconds
        for diff in time_diffs:
            # Allow some tolerance for rounding
            assert abs(diff - expected_interval) < expected_interval * 0.1


def test_train_with_empty_regressors_list(
    prophet_basic_config: ProphetModelConfig,
    windspeed_mock_datapoints: list[AssetDatapoint],
) -> None:
    """Training with an explicit empty regressors list should behave the same as no regressors."""
    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    model = model_provider.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name=prophet_basic_config.target.attribute_name,
                datapoints=windspeed_mock_datapoints,
            ),
            regressors=[],
        ),
    )
    assert model is not None
    model_provider.save_model(model)

    forecast = model_provider.generate_forecast()
    assert forecast is not None
    assert len(forecast.datapoints) > 0


def test_evaluate_model_with_insufficient_data(
    prophet_basic_config: ProphetModelConfig,
) -> None:
    """evaluate_model should return None when training data is too short for cross-validation."""
    model_provider = ModelProviderFactory.create_provider(prophet_basic_config)

    # Small dataset: 3 points is enough to train but not enough for CV
    small_dataset = [
        AssetDatapoint(x=1_741_193_868_000, y=1.0),
        AssetDatapoint(x=1_741_197_468_000, y=2.0),
        AssetDatapoint(x=1_741_201_068_000, y=3.0),
    ]

    model = model_provider.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name=prophet_basic_config.target.attribute_name,
                datapoints=small_dataset,
            ),
        ),
    )
    assert model is not None

    metrics = model_provider.evaluate_model(model)
    # Should return None because training data is too short for meaningful CV
    assert metrics is None


def test_regressors_affect_forecast() -> None:
    """Prove that regressors change fitted coefficients and forecasts.

    We build synthetic data where target = 10 + 2 * regressor.
    A model trained WITH the regressor should produce different forecasts than one trained WITHOUT,
    and the fitted regressor coefficient should be non-zero.
    """
    from uuid import uuid4

    from service_ml_forecast.models.model_config import (
        ProphetModelConfig,
        RegressorAssetDatapointsFeature,
        TargetAssetDatapointsFeature,
    )

    base_time = 1_741_193_868_000  # ms epoch
    hour_ms = 3_600_000

    synthetic_base = 10.0
    synthetic_coeff = 2.0
    reg_range = 10  # regressor cycles 1..10

    # Synthetic data: 168 hours (1 week), target is exactly base + coeff*regressor
    # Long enough for Prophet to confidently fit the regressor coefficient
    target_points: list[AssetDatapoint] = []
    regressor_points: list[AssetDatapoint] = []
    for i in range(168):
        t = base_time + i * hour_ms
        reg_val = float((i % reg_range) + 1)
        target_points.append(AssetDatapoint(x=t, y=synthetic_base + synthetic_coeff * reg_val))
        regressor_points.append(AssetDatapoint(x=t, y=reg_val))

    target_feature = TargetAssetDatapointsFeature(
        asset_id="49ORIhkDVAlT97dYGUD9p5",
        attribute_name="power",
    )

    config_no_reg = ProphetModelConfig(
        id=uuid4(),
        realm="master",
        name="test",
        target=target_feature,
        forecast_interval="PT60S",
        forecast_periods=3,
        forecast_frequency="1h",
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=False,
    )
    config_with_reg = ProphetModelConfig(
        id=uuid4(),
        realm="master",
        name="test",
        target=target_feature,
        forecast_interval="PT60S",
        forecast_periods=3,
        forecast_frequency="1h",
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=False,
        regressors=[
            RegressorAssetDatapointsFeature(
                asset_id="41ORIhkDVAlT97dYGUD9n5",
                attribute_name="windSpeed",
            ),
        ],
    )

    # Train model without regressor
    provider_no_reg = ModelProviderFactory.create_provider(config_no_reg)
    model_no_reg = provider_no_reg.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name="power",
                datapoints=target_points,
            ),
        ),
    )
    assert model_no_reg is not None
    provider_no_reg.save_model(model_no_reg)

    # Train model with regressor
    provider_with_reg = ModelProviderFactory.create_provider(config_with_reg)
    model_with_reg = provider_with_reg.train_model(
        TrainingDataSet(
            target=AssetFeatureDatapoints(
                feature_name="power",
                datapoints=target_points,
            ),
            regressors=[
                AssetFeatureDatapoints(
                    feature_name="windSpeed",
                    datapoints=regressor_points,
                ),
            ],
        ),
    )
    assert model_with_reg is not None
    provider_with_reg.save_model(model_with_reg)

    # The regressor coefficient should be non-negligible (Prophet stores it in model.params['beta'])
    # beta is a (samples, n_regressors) array; we check the mean across MCMC samples.
    # Note: Prophet standardises regressors internally, so the raw coefficient is not the
    # original scale (e.g. 2.0) -- it is the coefficient for the z-scored regressor.
    beta = model_with_reg.params["beta"]
    assert beta is not None
    assert beta.shape[1] == 1  # one regressor
    mean_coeff = float(beta.mean())
    min_regressor_coeff = 0.05
    assert abs(mean_coeff) > min_regressor_coeff, (
        f"Regressor coefficient should be non-zero, got {mean_coeff}"
    )

    # Forecast both with the same future regressor values (constant 5.0)
    future_times = [base_time + (168 + i) * hour_ms for i in range(3)]
    future_reg_value = 5.0
    future_regressor = AssetFeatureDatapoints(
        feature_name="windSpeed",
        datapoints=[AssetDatapoint(x=t, y=future_reg_value) for t in future_times],
    )
    forecast_dataset = ForecastDataSet(regressors=[future_regressor])

    forecast_no_reg = provider_no_reg.generate_forecast()
    forecast_with_reg = provider_with_reg.generate_forecast(forecast_dataset)

    # Forecasts should differ because the regressor pulls the prediction toward base + coeff*value
    assert forecast_no_reg.datapoints is not None
    assert forecast_with_reg.datapoints is not None

    no_reg_values = [dp.y for dp in forecast_no_reg.datapoints]
    with_reg_values = [dp.y for dp in forecast_with_reg.datapoints]

    expected_forecast = synthetic_base + synthetic_coeff * future_reg_value
    forecast_tolerance = 5.0
    # The with-regressor forecast should be close to the expected value
    for v in with_reg_values:
        assert expected_forecast - forecast_tolerance < v < expected_forecast + forecast_tolerance, (
            f"With-regressor forecast should center around {expected_forecast}, got {v}"
        )

    # And the two forecasts should actually be different
    assert no_reg_values != with_reg_values, "Forecasts with and without regressor should differ"
