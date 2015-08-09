"""
Receding Horizon Informative Exploration

Feature handling methods for receding horizon informative exploration
"""

class FeaturePack:

	def __init__(self, extractfn, whitenfn):

		self.extract = extractfn
		self.whiten = whitenfn