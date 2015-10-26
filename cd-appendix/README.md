# Informative Seafloor Exploration for Benthic Habitat Mapping

***Author***: Kelvin Y.S. Hsu


***Description***:

	A receding horizon approach to informative path planning using the linearised model differential entropy of Gaussian process classifiers
	This CD Appendix is provided purely for reference, and is not meant to serve as a stand alone software. The code is fairly documented, however documentation and further work is still under progress as of October 2015.


***Instructions***:

	-Please make sure the CD-Appendix is on the path of your computer
	-Besides the standard Python numerical libraries (numpy, scipy, etc), you must install 'nlopt' from 'http://ab-initio.mit.edu/wiki/index.php/NLopt'
	-Due to the large file sizes of the Scott Reef dataset, please contact the author separately for access to these datasets

	The main script which performs the informative path planning algorithm proposed in the paper and thesis is
		informative-seafloor-exploration/scott_reef_analysis.py

***Acknowledgements***:

	All work included in this Appendix is the independent work of the author, Kelvin Hsu, with the following exceptions.

		Collaborated Work: 

			Code:

				computers/unsupervised/
				computers/gp/{kernel.py, linalg.py, predict.py, train.py, types.py}

			Authors:

				Kelvin Y.S. Hsu
				Alistair Reid
				Simon O'Callaghan
				Lachlan McCalman
				Daniel Steinberg

	Specifically, the main author is the sole author to the following work.

		Independent Work:

			Code:

				computers/gp/classifier/
				computers/gp/partools.py
				concept-tests/
				figure-generation/
				informative-seafloor-exploration/
				sea/
				spatial-exploration/
				thesis-visuals/
				utils/

			Authors:

				Kelvin Y.S. Hsu


***Contact***:

	Please contact the author, Kelvin Hsu, if you have any questions regarding this CD Appendix.

	Email: yhsu9975@uni.sydney.edu.au OR Kelvin.Hsu@nicta.com.au
	GitHub Repository: https://github.com/KelvyHsu/ocean-exploration.git