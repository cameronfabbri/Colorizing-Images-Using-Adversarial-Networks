import cPickle as pickle

train = open('celeba_train.pkl', 'rb')
test  = open('celeba_test.pkl', 'rb')

train_data = pickle.load(train)
test_data  = pickle.load(test)

train_paths = []
for f in train_data:
   f = f.replace('/mnt/data2/images/celeba/images/','/home/fabbric/data/images/celeba/original/')
   train_paths.append(f)
pf   = open('new_celeba_train.pkl', 'wb')
data = pickle.dumps(train_paths)
pf.write(data)
pf.close()

test_paths = []
for f in test_data:
   f = f.replace('/mnt/data2/images/celeba/images/','/home/fabbric/data/images/celeba/original/')
   test_paths.append(f)
pf   = open('new_celeba_test.pkl', 'wb')
data = pickle.dumps(test_paths)
pf.write(data)
pf.close()

print len(train_data)
print len(test_data)
print len(train_paths)
print len(test_paths)

