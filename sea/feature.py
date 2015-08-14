"""
Informative Seafloor Exploration
"""
from scipy.spatial.distance import cdist

def closest_query_indices(Xq, Xq_ref):
	return cdist(Xq, Xq_ref).argmin(axis = 1)
	
def black_extract(Xq, Xq_ref, Fq_ref):
	iq_ref = closest_query_indices(Xq, Xq_ref)
	return Fq_ref[iq_ref]

def black_compose(Xq_ref, Fq_ref):

    def feature_fn(Xq):
    	return black_extract(Xq, Xq_ref, Fq_ref)
    feature_fn.Xq_ref = Xq_ref
    feature_fn.Fq_ref = Fq_ref
    return feature_fn

def white_extract(Xq, Xq_ref, Fq_ref, white_fn, white_params = None):
	iq_ref = closest_query_indices(Xq, Xq_ref)
	Fq = Fq_ref[iq_ref]
	return white_fn(Fq, white_params)

def white_compose(Xq_ref, Fq_ref, white_fn):

    def feature_fn(Xq, white_params = None):
    	return white_extract(Xq, Xq_ref, Fq_ref, white_fn, 
    		white_params = white_params)
    feature_fn.Xq_ref = Xq_ref
    feature_fn.Fq_ref = Fq_ref
    feature_fn.white_fn = white_fn
    return feature_fn

def extract(*args, **kwargs):
    
    if len(args) == 3:
        return black_extract(*args, **kwargs)
    else:
        return white_extract(*args, **kwargs)

def compose(*args):

    if len(args) == 2:
        return black_compose(*args)
    else:
        return white_compose(*args)

def black_fn(X, params = None):

    if params is None:
        return X, 0
    else:
        return X