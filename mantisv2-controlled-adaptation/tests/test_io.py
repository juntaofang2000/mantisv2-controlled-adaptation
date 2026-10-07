import numpy as np

from mantis_adaptation.io import safe_name, to_instance_list


def test_to_instance_list_univariate_2d():
    X = np.arange(24, dtype=np.float32).reshape(4, 6)
    instances = to_instance_list(X)
    assert len(instances) == 4
    assert instances[0].shape == (1, 6)


def test_to_instance_list_multivariate_3d():
    X = np.zeros((5, 3, 11), dtype=np.float32)
    instances = to_instance_list(X)
    assert len(instances) == 5
    assert instances[0].shape == (3, 11)


def test_safe_name():
    assert safe_name("A/B dataset") == "A_B_dataset"

