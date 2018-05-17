import numpy as np
import os
import pickle

from scipy.sparse import csr_matrix

class SparseFeatureLoader:
    def __init__(self, csr_data_path, preprocessed_data_path):
        self.__load_csr(csr_data_path)
        self.__load_operation_index(preprocessed_data_path)
        
    def __load_csr(self, path_to_csr_data):
        with open(os.path.join(path_to_csr_data, 'sparse_matrix_info.pkl'), 'rb') as handle:
            sparse_matrix_info = pickle.load(handle)
            
        data_memmap = np.memmap(os.path.join(path_to_csr_data, 'data.memmap'),
                                mode='r',
                                dtype=np.float32,
                                shape=(sparse_matrix_info['nnz'],))

        col_idx_memmap = np.memmap(os.path.join(path_to_csr_data, 'col_idx.memmap'),
                                    mode='r',
                                    dtype=np.int16,
                                    shape=(sparse_matrix_info['nnz'],))
        indptr_memmap = np.memmap(os.path.join(path_to_csr_data, 'indptr.memmap'),
                                mode='r',
                                dtype=np.int64,
                                shape=(sparse_matrix_info['shape'][0] + 1,))
        self.data = csr_matrix((data_memmap, col_idx_memmap, indptr_memmap), 
                               shape=sparse_matrix_info['shape'], dtype=data_memmap.dtype)
        
    def __load_operation_index(self, path_to_preprocessed_data):
        with open(os.path.join(path_to_preprocessed_data, 'year_operationid_counter.pkl'), 'rb') as handle:
            self.operationid_counter = pickle.load(handle)
            
    def get(self, operation_ids):
        operation_indices = [self.operationid_counter[op] for op in operation_ids]
        return self.data[operation_indices]