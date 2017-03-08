import cPickle as pickle
import sys
pkl_file = open(sys.argv[1], 'rb')
exp_info = pickle.load(pkl_file)
print exp_info
