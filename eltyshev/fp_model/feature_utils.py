import numpy as np
import os
import pickle

from scipy.sparse import csr_matrix
from scipy.sparse import load_npz

class SparseFeatureLoader:
    def __init__(self, data_path, memmap=False):
        if memmap:
            self.__load_memmap(data_path)
        else:
            self.__load_csr(data_path)
            
        self.__load_operation_index(data_path)
        self.__load_column_names(data_path)
        
    def __load_csr(self, data_path):
        self.data = load_npz(os.path.join(data_path, "data.npz"))
        
    def __load_memmap(self, data_path):
        with open(os.path.join(data_path, 'sparse_matrix_info.pkl'), 'rb') as handle:
            sparse_matrix_info = pickle.load(handle)
            
        data_memmap = np.memmap(os.path.join(data_path, 'data.memmap'),
                                mode='r',
                                dtype=np.float32,
                                shape=(sparse_matrix_info['nnz'],))

        col_idx_memmap = np.memmap(os.path.join(data_path, 'col_idx.memmap'),
                                    mode='r',
                                    dtype=np.int16,
                                    shape=(sparse_matrix_info['nnz'],))
        indptr_memmap = np.memmap(os.path.join(data_path, 'indptr.memmap'),
                                mode='r',
                                dtype=np.int64,
                                shape=(sparse_matrix_info['shape'][0] + 1,))
        self.data = csr_matrix((data_memmap, col_idx_memmap, indptr_memmap), 
                               shape=sparse_matrix_info['shape'], dtype=data_memmap.dtype)
        
    def __load_operation_index(self, data_path):
        with open(os.path.join(data_path, 'operationid_counter.pkl'), 'rb') as handle:
            self.operationid_counter = pickle.load(handle)
            
    def __load_column_names(self, data_path):
        with open(os.path.join(data_path, 'column_names.pkl'), 'rb') as handle:
            self.columns = pickle.load(handle)
            
    def get(self, operation_ids):
        operation_indices = [self.operationid_counter[op] for op in operation_ids if op in self.operationid_counter]
        return self.data[operation_indices]
    
    def have_features_for_ids(self, operation_ids):
        return np.array([op in self.operationid_counter for op in operation_ids])