from unittest.mock import Mock, patch

from torch import Tensor, rand

from wsmf.metamodels.networks import Dataset2VecMetricLearning


@patch("dataset2vec.Dataset2Vec.forward")
def test_d2v_metric_forward(dataset2vec_mock: Mock) -> None:
    # Given
    dataset2vec_mock.return_value = Tensor([1.0, 2.0, 3.0])
    meta_model = Dataset2VecMetricLearning()
    X, y = rand((10, 5)), rand(10, 1)

    # When
    actual_encoding = meta_model(X, y)

    # Then
    assert (actual_encoding == Tensor([1.0, 2.0, 3.0])).all()
