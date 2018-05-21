import numpy as np
import os
import pickle

from scipy.sparse import csr_matrix
from scipy.sparse import load_npz


class SparseFeatureLoader:
    def __init__(self, data_path, memmap=True):
        if memmap:
            self.__load_memmap(data_path)
        else:
            self.__load_csr(data_path)
        self.data_path = data_path

        self.__load_operation_index(data_path)
        self.__load_column_names(data_path)

    def __load_csr(self, data_path):
        self.data = load_npz(os.path.join(data_path, "data.npz"))

    def __load_memmap(self, data_path):
        with open(os.path.join(data_path, 'sparse_matrix_info.pkl'), 'rb') as handle:
            sparse_matrix_info = pickle.load(handle)

        self.data_memmap = np.memmap(os.path.join(data_path, 'data.memmap'),
                                     mode='r',
                                     dtype=np.float32,
                                     shape=(sparse_matrix_info['nnz'],))

        self.col_idx_memmap = np.memmap(os.path.join(data_path, 'col_idx.memmap'),
                                        mode='r',
                                        dtype=np.int16,
                                        shape=(sparse_matrix_info['nnz'],))
        self.indptr_memmap = np.memmap(os.path.join(data_path, 'indptr.memmap'),
                                       mode='r',
                                       dtype=np.int64,
                                       shape=(sparse_matrix_info['shape'][0] + 1,))

    def __load_operation_index(self, data_path):
        with open(os.path.join(data_path, 'operationid_counter.pkl'), 'rb') as handle:
            self.operationid_counter = pickle.load(handle)

    def __load_column_names(self, data_path):
        with open(os.path.join(data_path, 'column_names.pkl'), 'rb') as handle:
            self.columns = pickle.load(handle)

    def _get(self, indices_to_get):
        with open(os.path.join(self.data_path, 'sparse_matrix_info.pkl'), 'rb') as handle:
            sparse_matrix_info = pickle.load(handle)

        if np.isscalar(indices_to_get):
            indices_to_get = [indices_to_get]
        indices_to_get = np.array(indices_to_get, dtype=int)

        indices_delta = np.zeros(len(indices_to_get) + 1, dtype=self.indptr_memmap.dtype)
        indices_delta[1:] = self.indptr_memmap[indices_to_get + 1] - self.indptr_memmap[indices_to_get]

        new_indptr = np.cumsum(indices_delta)
        new_data = np.zeros(new_indptr[-1], dtype=self.data_memmap.dtype)
        new_col_idx = np.zeros(new_indptr[-1], dtype=self.col_idx_memmap.dtype)

        for new_index, index_to_get in enumerate(indices_to_get):
            new_data[new_indptr[new_index]: new_indptr[new_index + 1]] = \
                self.data_memmap[self.indptr_memmap[index_to_get]:self.indptr_memmap[index_to_get + 1]]
            new_col_idx[new_indptr[new_index]: new_indptr[new_index + 1]] = \
                self.col_idx_memmap[self.indptr_memmap[index_to_get]:self.indptr_memmap[index_to_get + 1]]

        return csr_matrix((new_data, new_col_idx, new_indptr),
                          shape=(len(indices_to_get), sparse_matrix_info['shape'][1]), dtype=new_data.dtype)

    def get(self, operation_ids):
        operation_indices = [self.operationid_counter[op] for op in operation_ids if op in self.operationid_counter]
        return self._get(operation_indices)

    def have_features_for_ids(self, operation_ids):
        return np.array([op in self.operationid_counter for op in operation_ids])