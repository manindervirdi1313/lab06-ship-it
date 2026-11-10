"""Tests for the late-delivery prediction engine."""

import pytest

import predict


@pytest.fixture
def valid_order():
    """Provide valid delivery inputs."""
    return {
        "distance_km": 5,
        "items": 3,
        "hour": 18,
        "prep_minutes": 20,
        "courier_months": 12,
        "weather": "Clear",
        "day_type": "Weekday",
        "vehicle": "Bike",
    }


def test_prediction_returns_expected_fields(valid_order):
    result = predict.predict_order(**valid_order)

    assert "late_probability" in result
    assert "late_percentage" in result
    assert "prediction" in result
    assert "voucher_decision" in result
    assert "threshold" in result


def test_probability_is_between_zero_and_one(valid_order):
    result = predict.predict_order(**valid_order)

    assert 0 <= result["late_probability"] <= 1


def test_voucher_decision_uses_threshold(valid_order):
    result = predict.predict_order(**valid_order)

    if result["late_probability"] >= 0.30:
        assert result["voucher_decision"] == "Issue voucher"
    else:
        assert result["voucher_decision"] == "No voucher"


def test_missing_preparation_time_is_supported(valid_order):
    valid_order["prep_minutes"] = None

    result = predict.predict_order(**valid_order)

    assert 0 <= result["late_probability"] <= 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_km", -5),
        ("distance_km", 0),
        ("items", 0),
        ("items", 2.5),
        ("hour", 25),
        ("courier_months", -1),
        ("prep_minutes", -5),
        ("weather", ""),
        ("day_type", ""),
        ("vehicle", ""),
    ],
)
def test_invalid_inputs_raise_error(valid_order, field, value):
    valid_order[field] = value

    with pytest.raises(ValueError):
        predict.predict_order(**valid_order)


@pytest.mark.parametrize(
    ("probability", "expected_decision"),
    [
        (0.29, "No voucher"),
        (0.30, "Issue voucher"),
        (0.75, "Issue voucher"),
    ],
)
def test_voucher_threshold_boundary(
    valid_order,
    monkeypatch,
    probability,
    expected_decision,
):
    class FakeModel:
        classes_ = [0, 1]

        def predict_proba(self, data):
            return [[1 - probability, probability]]

    monkeypatch.setattr(predict, "model", FakeModel())

    result = predict.predict_order(**valid_order)

    assert result["voucher_decision"] == expected_decision
