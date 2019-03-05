import caffe
import lmdb

path     = '/home/barna/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb'
lmdb_env = lmdb.open(path, readonly=True, lock=True)
# lmdb_env    = lmdb.open('/home/barna/data/VOCdevkit/Custom/lmdb/test_lmdb')

count = 0