import numpy as np
import pytest

from mantis_adaptation.protocol import Protocol, scheduled_learning_rate


def test_gradient_accumulation_metadata():
    protocol = Protocol(batch_size=128, micro_batch_size=4)
    protocol.validate()
    assert protocol.gradient_accumulation_steps == 32
    assert protocol.to_dict()["effective_batch_size"] == 128


def test_invalid_micro_batch_size():
    with pytest.raises(ValueError, match="divisible"):
        Protocol(batch_size=128, micro_batch_size=3).validate()


def test_schedule_matches_official_update_semantics():
    first = scheduled_learning_rate(
        epochs=100,
        updates_per_epoch=5,
        update_step=1,
        base_learning_rate=2e-4,
    )
    warmup_end = scheduled_learning_rate(
        epochs=100,
        updates_per_epoch=5,
        update_step=50,
        base_learning_rate=2e-4,
    )
    final = scheduled_learning_rate(
        epochs=100,
        updates_per_epoch=5,
        update_step=500,
        base_learning_rate=2e-4,
    )
    assert np.isclose(first, 4e-6)
    assert np.isclose(warmup_end, 2e-4)
    assert np.isclose(final, 0.0, atol=1e-15)
