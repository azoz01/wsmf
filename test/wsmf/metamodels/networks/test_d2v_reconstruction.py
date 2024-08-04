from unittest.mock import Mock, patch

from torch import Size, Tensor, rand

from wsmf.metamodels.networks import Dataset2VecForLandmarkerReconstruction


def test_meta_model_returns_output_of_proper_dimensionality() -> None:
    # Given
    meta_model = Dataset2VecForLandmarkerReconstruction(4)
    X, y = rand((10, 5)), rand((10, 1))

    # When
    reconstructed_landmarkers = meta_model(X, y)

    # Then
    assert reconstructed_landmarkers.shape == Size([4])


@patch("wsmf.metamodels.networks.d2v_reconstruction.roc_auc_activation")
def test_meta_model_uses_reconstructor(roc_auc_activation_mock: Mock) -> None:
    # Given
    roc_auc_activation_mock.return_value = Tensor([4, 5, 6])
    meta_model = Dataset2VecForLandmarkerReconstruction(3)
    encoder_mock = Mock(return_value=Tensor([1, 2, 3]))
    meta_model.dataset2vec.forward = encoder_mock
    reconstructor_mock = Mock(return_value=Tensor([4, 5, 6]))
    meta_model.landmarker_reconstructor.forward = reconstructor_mock  # type: ignore # noqa: E501
    X, y = rand((10, 5)), rand((10, 1))

    # When
    reconstructed_landmarkers = meta_model(X, y)

    # Then
    encoder_calls = encoder_mock.call_args
    reconstructor_calls = reconstructor_mock.call_args

    assert (reconstructed_landmarkers == Tensor([4, 5, 6])).all()
    assert (encoder_calls[0][0] == X).all()
    assert (encoder_calls[0][1] == y).all()
    assert (reconstructor_calls[0][0] == Tensor([1, 2, 3])).all()
