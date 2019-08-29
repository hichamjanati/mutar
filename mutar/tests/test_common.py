import pytest

from sklearn.utils.estimator_checks import check_estimator

from mutar import DirtyModel

# @pytest.mark.parametrize(
#     "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
# )
# def test_all_estimators(Estimator):
#     return check_estimator(Estimator)
