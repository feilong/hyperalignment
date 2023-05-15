from .procrustes import procrustes
from .searchlight import compute_searchlight_weights as searchlight_weights
from .searchlight import searchlight_hyperalignment, searchlight_procrustes, searchlight_ridge, searchlight_template
from .ridge import ridge, ridge_grid, ensemble_ridge
from .local_template import compute_template
from .ensemble import compute_ensemble_indices
from .sparse import initialize_sparse_matrix
