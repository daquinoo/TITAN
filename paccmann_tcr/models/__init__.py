from .knn import knn
from .titan_multiscale import TITANMultiScale

# More models could follow
MODEL_FACTORY = {
    'knn': knn
}
