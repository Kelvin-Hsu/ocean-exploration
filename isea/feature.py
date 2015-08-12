"""
Receding Horizon Informative Exploration

Feature handling methods for receding horizon informative exploration
"""

def closest_query_indices(Xq, Xq_ref):
	return cdist(Xq, Xq_ref).argmin(axis = 1)
	
def extract_feature(Xq, Xq_ref, Fq_ref):
	iq_ref = closest_query_indices(Xq, Xq_ref)
	return Fq_ref[iq_ref]

def compose_feature_fn(Xq_ref, Fq_ref):

    def feature_fn(Xq):
    	return extract_feature(Xq, Xq_ref, Fq_ref)
    feature_fn.Xq_ref = Xq_ref
    feature_fn.Fq_ref = Fq_ref
    return feature_fn

def extract_white_feature(Xq, Xq_ref, Fq_ref, white_fn, white_params = None):
	iq_ref = closest_query_indices(Xq, Xq_ref)
	Fq = Fq_ref[iq_ref]
	return white_fn(Fq, white_params = white_params)

def compose_white_feature_fn(Xq_ref, Fq_ref, white_fn):

    def feature_fn(Xq, white_params = None):
    	return extract_white_feature(Xq, Xq_ref, Fq_ref, white_fn, 
    		white_params = white_params)
    feature_fn.Xq_ref = Xq_ref
    feature_fn.Fq_ref = Fq_ref
    feature_fn.white_fn = white_fn
    return feature_fn


