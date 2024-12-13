from .knn import knn
from .titan_multiscale import TITANMultiScale
from .paccmann_predictor.models import BimodalMCA

# More models could follow
MODEL_FACTORY = {
    'knn': knn,
    'bimodal_mca': BimodalMCA,
    'bimodal_mca_multiscale': TITANMultiScale
}
