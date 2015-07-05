#!/usr/bin/python
"""
command line script 'rename'

--Usage--
python rename.py filename newfilename oldfieldname newfieldname

or

python rename.py filename oldfieldname newfieldname

--Description--

	Renames a key/field in 'filename.npz' from 'oldfieldname' to 'newfieldname'
	Stores the new file in 'newfilename.npz'

	If 'newfilename' is ignored, the new file name is simply the old file name
	with '_new' attached.
---------------
"""

import numpy as np
import sys

# Note that file name is always the first command line argument
narg = len(sys.argv)

if narg == 4:
	
	filename = sys.argv[1]
	if '.' not in filename:
		newfilename = filename + '_new.npz'
		filename += '.npz'
	else:
		if filename.split('.')[1] != 'npz':
			raise ValueError('"%s" is not a .npz file.' % filename)
		newfilename = filename.split('.')[0] + '_new.npz'

	old_name = sys.argv[2]
	new_name = sys.argv[3]

elif narg == 5:

	filename = sys.argv[1]
	if '.' not in filename:
		filename += '.npz'
	newfilename = sys.argv[2]
	if '.' not in newfilename:
		newfilename += '.npz'
	old_name = sys.argv[3]
	new_name = sys.argv[4]	

else:

	if narg < 4:
		raise ValueError('Too little arguments!')
	else:
		raise ValueError('Too many arguments!')

print('Loading "%s"...' % filename)
data = np.load(filename)

data_dict_new = dict()

print('Copying field data from "%s" to "%s"...' % (filename, newfilename))
for key in data.keys():

	if key == old_name:
		data_dict_new.update({new_name: data[old_name]})
		print('\tField "%s" replaced by "%s".' % (old_name, new_name))
	else:
		data_dict_new.update({key: data[key]})
		print('\tField "%s" copied.' % key)

data.close()
print('Saving renamed fields into "%s"...' % newfilename)
np.savez(newfilename, **data_dict_new)
print('Done')

