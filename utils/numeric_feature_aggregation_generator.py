import os
import sys
import pickle
import numpy as np
import scipy as sp
import pandas as pd

import tqdm
import scipy.sparse
from utils import Timer
from functools import partial
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from preprocessing import read_tables

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


path_to_dataset = '/home/shared_files/alfa_half_year/'
path_to_local_temp_data = os.path.join(path_to_dataset, 'tmp')
path_to_graph_data = os.path.join(path_to_dataset, 'transaction_graph')
path_to_preprocessed_data = os.path.join(path_to_dataset, 'preprocessed')
path_to_generated_features = os.path.join(path_to_dataset, 'generated_features')
path_to_generated_baseamount_agg = os.path.join(path_to_generated_features, 'baseamount_agg')
path_to_generated_links_agg = os.path.join(path_to_generated_features, 'links_agg')


DEFAULT_LOCAL_TEMP_DATA_DIRNAME = 'tmp'
DEFAULT_GRAPH_DATA_DIRNAME = 'transaction_graph'
DEFAULT_PREPROCESSED_DATA_DIRNAME = 'preprocessed'
DEFAULT_GENERATED_FEATURES_DIRNAME = 'generated_features'
DEFAULT_GENERATED_BASEAMOUNT_AGG_DIRNAME = 'baseamount_agg'
DEFAULT_GENERATED_LINKS_AGG_DIRNAME = 'links_agg'

off_ops_filename = 'off_ops.csv'
off_members_filename = 'off_members.csv'
susp_ops_filename = 'susp_ops.csv'
susp_members_filename = 'susp_members.csv'

def concat_files(filenames, out_filename):
    with open(out_filename, 'w') as write_handle:
        for filename in filenames:
            for line in open(filename, 'r'):
                write_handle.write(line)


def fill_zeros_with_last(arr):
    """
    Fill all zeros in array with leftmost nonzero element for each zero subsequence

    Parameters
    ----------
    a : array_like
        An array to fill zeros in

    Returns
    -------
    filled : ndarray
        An array of same size as a, with zeros filled

    Notes
    -----
    Leftmost zeros will remain zeros because there is no leftmost nonzero element for them
    """
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def numeric_feature_generate_chunk(tupled_data,
                                   global_delta=0,
                                   stats_to_get=['mean', 'count'],
                                   backward=[86400 * 7, 86400 * 3, 86400 * 1]):
    """
    Generate numeric features through categorical columns slices for some chunk of operation indices

    Parameters
    ----------
    tupled_data : tuple
        Tupled arguments that contains:
            second_clientid_opid_chunk : ndarray
                An array of shape (chunk_size, 3) that contains
                [second, client_id, operation_id] in every row
            flat_stat : dict
                Dictionary containing helper subdicts with
                sparse matrices for each column_to_slice_by
            numeric_csr : csr_matrix of shape (n_operations, n_clients)
                Sparse matrix containing information about
                operation-client-numeric_feature links
            sec_to_op_index : ndarray of shape (max(seconds_from_start) + 1,)
                An array that maps every seconds into id
                of first operation that occurs at second.
            columns_to_slice_by: iterable
                An iterable contains categorical columns names
                for which slices feature generation performs
            filename: str
                Name of file to write triplets (i, j, value)
    global_delta : int, optional
        Global column offset of newly generated features
    stats_to_get : list of ['mean', 'std', 'count'], optional
        List of aggregates to compute
    backward : list of ints, optional
        Amounts of seconds to look back when building aggregates.
        Aggregates will be generated for every backward from backwards

    Returns
    -------
    nnz : int
        number of non-zero entries wrote to disk

    Notes
    -----
    numeric_feature_generate_chunk produce csv-like file filled with triplets
    (row_idx, col_idx, data) for every file row.
    """
    second_clientid_opid_chunk, flat_stat, numeric_csr, sec_to_op_index, columns_to_slice_by, filename = tupled_data
    st_to_func = {
        'mean': np.mean,
        'std': np.std,
        'count': len
    }

    deltas = [0]
    for slice_by in columns_to_slice_by:
        stat = flat_stat[slice_by]
        for st in stats_to_get:
            deltas.append(deltas[-1] + len(stat['slice_encoder']))
    deltas = [deltas]
    for _ in range(1, len(backward)):
        deltas.append([deltas[-1][-1] + el for el in deltas[0]])

    #print(second_clientid_opid_chunk, flush=True)
    #print(second_clientid_opid_chunk[0][0] - backward[0], max(second_clientid_opid_chunk[0][0] - backward[0], 0))
    chunk_from = sec_to_op_index[max(second_clientid_opid_chunk[0][0] - backward[0], 0)]
    chunk_to = sec_to_op_index[second_clientid_opid_chunk[-1][0]]
    chunk_numeric_csc = numeric_csr[chunk_from:chunk_to].tocsc()
    chunk_flat_stat = {}
    for slice_by in columns_to_slice_by:
        chunk_flat_stat[slice_by] = {'mat_csc': flat_stat[slice_by]['mat_csr'][chunk_from: chunk_to].tocsc()}

    nnz = 0

    with open(filename, 'w') as handle:
        for second_clientid_opid in second_clientid_opid_chunk:
            second, client_id, op_id = second_clientid_opid
            for bw_idx, bw in enumerate(backward):
                sec_back = max(second - bw, 0)
                i_from = sec_to_op_index[sec_back] - chunk_from
                i_to = op_id - chunk_from

                if i_from > i_to:
                    continue
                sliced_numeric = chunk_numeric_csc[i_from: i_to, client_id]
                if sliced_numeric.nnz == 0:
                    continue

                sliced_numeric_data = sliced_numeric.data
                for i, slice_by in enumerate(columns_to_slice_by):
                    stat = flat_stat[slice_by]
                    max_shift = len(stat['slice_encoder'])
                    chunk_stat = chunk_flat_stat[slice_by]

                    sliced_category = chunk_stat['mat_csc'][i_from:i_to, client_id]
                    sliced_category_data = sliced_category.data

                    numeric_data_by_category = {
                        uniq: sliced_numeric_data[sliced_category_data == uniq]
                        for uniq in np.unique(sliced_category_data)
                    }
                    for j, st in enumerate(stats_to_get):
                        for uniq, numeric_data in numeric_data_by_category.items():
                            handle.write('{}\t{}\t{}\n'.format(op_id,
                                                               global_delta + deltas[bw_idx][
                                                                   i * len(stats_to_get) + j] + uniq - 1,
                                                               st_to_func[st](numeric_data)))
                            nnz += 1
    return nnz


class PerOperationNumericFeatureAggregationGenerator():
    """
    Class for customizable generation of per-operation numeric features aggregates through selected categorical features slices
    """

    def __init__(self, path_to_dataset, **kwargs):
        self.path_to_dataset = path_to_dataset
        self.path_to_preprocessed_data = os.path.join(
            path_to_dataset,
            kwargs.get('preprocessed_data_dirname', DEFAULT_PREPROCESSED_DATA_DIRNAME)
        )
        self.path_to_graph_data = os.path.join(
            path_to_dataset,
            kwargs.get('graph_data_dirname', DEFAULT_GRAPH_DATA_DIRNAME)
        )
        self.path_to_local_temp_data = os.path.join(
            path_to_dataset,
            kwargs.get('local_temp_data_dirname', DEFAULT_LOCAL_TEMP_DATA_DIRNAME)
        )
        self.path_generated_features = os.path.join(
            path_to_dataset,
            kwargs.get('generated_features_dirname', DEFAULT_GENERATED_FEATURES_DIRNAME)
        )
        with open(os.path.join(self.path_to_graph_data, 'graph_trans_df.pkl'), 'rb') as handle:
            self.graph_trans_df = pickle.load(handle)

        with open(os.path.join(self.path_to_preprocessed_data, 'clientid_counter.pkl'), 'rb') as handle:
            self.clientid_counter = pickle.load(handle)

        sorted_seconds_uniqs = np.sort(self.graph_trans_df[('all', 'seconds_from_start')].unique(),
                                       kind='mergesort').astype(int)
        self.sec_to_op_index = np.zeros(int(self.graph_trans_df[('all', 'seconds_from_start')].max()) + 1, dtype=int)
        sec_uniqs, sec_op_ids = np.unique(self.graph_trans_df[('all', 'seconds_from_start')].values.astype(int),
                                          return_index=True)

        self.sec_to_op_index[sec_uniqs] = sec_op_ids
        self.sec_to_op_index = fill_zeros_with_last(self.sec_to_op_index).astype(int)

    @property
    def n_operations(self):
        return len(self.graph_trans_df)

    @property
    def n_clients(self):
        return len(self.clientid_counter)

    def generate_helper_dicts_to_slice_by(self, columns_to_slice_by):
        # generate helper sparse slice matrices
        source_trans = self.graph_trans_df[~self.graph_trans_df[1, 'P_CLIENTID'].isnull()]
        target_trans = self.graph_trans_df[~self.graph_trans_df[2, 'P_CLIENTID'].isnull()]

        source_flat_stat = {}
        target_flat_stat = {}
        for slice_by in columns_to_slice_by:
            slice_uniques = self.graph_trans_df[slice_by].unique()
            slice_encoder = {v: i for v, i in zip(slice_uniques, np.arange(1, len(slice_uniques) + 1))}

            source_stat = {}
            target_stat = {}

            source_stat['slice_encoder'] = slice_encoder
            target_stat['slice_encoder'] = slice_encoder

            source_stat['mat'] = coo_matrix((source_trans[slice_by].map(slice_encoder.__getitem__),
                                             (source_trans['all', 'ID'],
                                              source_trans[1, 'P_CLIENTID'].values.astype(int))),
                                            shape=(self.n_operations, self.n_clients),
                                            dtype=np.int)
            target_stat['mat'] = coo_matrix((target_trans[slice_by].map(slice_encoder.__getitem__),
                                             (target_trans['all', 'ID'],
                                              target_trans[2, 'P_CLIENTID'].values.astype(int))),
                                            shape=(self.n_operations, self.n_clients),
                                            dtype=np.int)

            source_stat['mat_csr'] = source_stat['mat'].tocsr()
            source_stat['mat_csc'] = source_stat['mat'].tocsc()
            target_stat['mat_csr'] = target_stat['mat'].tocsr()
            target_stat['mat_csc'] = target_stat['mat'].tocsc()

            source_flat_stat[slice_by] = source_stat
            target_flat_stat[slice_by] = target_stat
        return source_flat_stat, target_flat_stat

    def generate_helper_dicts_to_numeric_feature(self, numeric_colname):
        source_trans = self.graph_trans_df[~self.graph_trans_df[1, 'P_CLIENTID'].isnull()]
        target_trans = self.graph_trans_df[~self.graph_trans_df[2, 'P_CLIENTID'].isnull()]
        source_numeric = coo_matrix((source_trans[numeric_colname],
                                     (source_trans['all', 'ID'], source_trans[1, 'P_CLIENTID'].values.astype(int))),
                                    shape=(self.n_operations, self.n_clients),
                                    dtype=np.float32)
        target_numeric = coo_matrix((target_trans[numeric_colname],
                                     (target_trans['all', 'ID'], target_trans[2, 'P_CLIENTID'].values.astype(int))),
                                    shape=(self.n_operations, self.n_clients),
                                    dtype=np.float32)
        return source_numeric, target_numeric

    def generate_colnames(self, flat_stat, columns_to_slice_by, backward, stats_to_get):
        colnames = []
        for slice_by in columns_to_slice_by:
            for bw in backward:
                stat = flat_stat[slice_by]
                slice_encoder = stat['slice_encoder']
                slice_decoder = {v: k for k, v in slice_encoder.items()}
                for st in stats_to_get:
                    colnames += [f'{slice_by[1]}:{bw // 86400}d:{st}:{slice_decoder[encoded]}' for encoded in
                                 np.arange(1, len(slice_encoder) + 1)]
        return colnames

    def generate_features(self,
                          numeric_colname,
                          columns_to_slice_by,
                          numeric_agg_dir,
                          stats_to_get=['mean', 'count'],
                          backward=[86400 * 7, 86400 * 3, 86400 * 1],
                          chunk_size=20000,
                          n_jobs=1,
                          verbose=1):
        source_flat_stat, target_flat_stat = self.generate_helper_dicts_to_slice_by(columns_to_slice_by)
        source_numeric, target_numeric = self.generate_helper_dicts_to_numeric_feature(numeric_colname)

        source_numeric_csr = source_numeric.tocsr()
        target_numeric_csr = target_numeric.tocsr()

        # prepare data for feature generation

        if_source = ~self.graph_trans_df[1, 'P_CLIENTID'].isnull()
        source_tqdm_max_ = if_source.sum()

        source_seconds = self.graph_trans_df[('all', 'seconds_from_start')][if_source].values.astype(int)
        source_indices = self.graph_trans_df[(1, 'P_CLIENTID')][if_source].values.astype(int)
        source_op_indices = self.graph_trans_df[('all', 'ID')][if_source].values.astype(int)

        source_pool_input = np.c_[source_seconds, source_indices, source_op_indices]
        source_n_chunks = len(source_pool_input) // chunk_size + (
            1 if len(source_pool_input) % chunk_size != 0 else 0)
        source_pool_input_chunks = [
            source_pool_input[i * chunk_size: min((i + 1) * chunk_size, len(source_pool_input))]
            for i in range(source_n_chunks)
        ]

        if_target = ~self.graph_trans_df[1, 'P_CLIENTID'].isnull()
        target_tqdm_max_ = if_source.sum()

        target_seconds = self.graph_trans_df[('all', 'seconds_from_start')][if_source].values.astype(int)
        target_indices = self.graph_trans_df[(1, 'P_CLIENTID')][if_source].values.astype(int)
        target_op_indices = self.graph_trans_df[('all', 'ID')][if_source].values.astype(int)

        target_pool_input = np.c_[target_seconds, target_indices, target_op_indices]
        target_n_chunks = len(target_pool_input) // chunk_size + (
            1 if len(target_pool_input) % chunk_size != 0 else 0)
        target_pool_input_chunks = [
            target_pool_input[i * chunk_size: min((i + 1) * chunk_size, len(target_pool_input))]
            for i in range(target_n_chunks)
        ]

        # start feature generation
        source_out_filenames = [
            os.path.join(self.path_to_local_temp_data,
                         f'source_{numeric_colname}_out_{i * chunk_size}-{min(len(source_pool_input), (i + 1) * chunk_size) - 1}')
            for i in range(source_n_chunks)
        ]
        source_out_zip = zip(source_pool_input_chunks,
                             [source_flat_stat] * source_n_chunks,
                             [source_numeric_csr] * source_n_chunks,
                             [self.sec_to_op_index] * source_n_chunks,
                             [columns_to_slice_by] * source_n_chunks,
                             source_out_filenames)

        source_in_filenames = [
            os.path.join(self.path_to_local_temp_data,
                         f'source_{numeric_colname}_in_{i * chunk_size}-{min(len(source_pool_input), (i + 1) * chunk_size) - 1}')
            for i in range(source_n_chunks)
        ]
        source_in_zip = zip(source_pool_input_chunks,
                            [target_flat_stat] * source_n_chunks,
                            [target_numeric_csr] * source_n_chunks,
                            [self.sec_to_op_index] * source_n_chunks,
                            [columns_to_slice_by] * source_n_chunks,
                            source_in_filenames)

        target_out_filenames = [
            os.path.join(self.path_to_local_temp_data,
                         f'source_{numeric_colname}_out_{i * chunk_size}-{min(len(target_pool_input), (i + 1) * chunk_size) - 1}')
            for i in range(target_n_chunks)
        ]
        target_out_zip = zip(target_pool_input_chunks,
                             [source_flat_stat] * target_n_chunks,
                             [source_numeric_csr] * target_n_chunks,
                             [self.sec_to_op_index] * target_n_chunks,
                             [columns_to_slice_by] * target_n_chunks,
                             target_out_filenames)

        target_in_filenames = [
            os.path.join(self.path_to_local_temp_data,
                         f'source_{numeric_colname}_in_{i * chunk_size}-{min(len(target_pool_input), (i + 1) * chunk_size) - 1}')
            for i in range(target_n_chunks)
        ]
        target_in_zip = zip(target_pool_input_chunks,
                            [target_flat_stat] * target_n_chunks,
                            [target_numeric_csr] * target_n_chunks,
                            [self.sec_to_op_index] * target_n_chunks,
                            [columns_to_slice_by] * target_n_chunks,
                            target_in_filenames)

        global_delta = 0
        nnz_counts = []
        n_proc = n_jobs if n_jobs > 0 else cpu_count() - n_jobs + 1
        # for tupled_zip, total_len in list(zip([source_out_zip, source_in_zip, target_out_zip, target_in_zip],
        #                                       [source_n_chunks] * 2 + [target_n_chunks] * 2)):
        #     nnz_counts.append(
        #         list(
        #             tqdm.tqdm(
        #                 map(
        #                     partial(numeric_feature_generate_chunk, backward=backward, global_delta=global_delta),
        #                     tupled_zip
        #                 ),
        #                 total=total_len,
        #                 disable=(verbose == 0)
        #             )
        #         )
        #     )


        with Pool(processes=n_proc) as pool:
            for tupled_zip, total_len in list(zip([source_out_zip, source_in_zip, target_out_zip, target_in_zip],
                                             [source_n_chunks] * 2 + [target_n_chunks] * 2)):
                nnz_counts += list(
                    tqdm.tqdm(
                        pool.imap_unordered(
                            partial(numeric_feature_generate_chunk, backward=backward, global_delta=global_delta),
                            tupled_zip
                        ),
                        total=total_len,
                        disable=(verbose == 0)
                    )
                )
        nnz = sum(nnz_counts)
        colnames_base = self.generate_colnames(source_flat_stat, columns_to_slice_by, backward, stats_to_get)
        colnames = []
        for prefix in ['source:out:', 'source:in', 'target:out', 'target:in']:
            colnames += [prefix + colname_base for colname_base in colnames_base]
        colnames = np.array(colnames)

        path_to_numeric_agg = os.path.join(self.path_generated_features, numeric_agg_dir)
        coo_filename = os.path.join(path_to_numeric_agg, 'coo.csv')
        concat_files(source_out_filenames + source_in_filenames + target_out_filenames + target_in_filenames,
                     os.path.join(self.path_generated_features, numeric_agg_dir, 'coo.csv'))
        with open(os.path.join(numeric_agg_dir, 'column_names.pkl'), 'wb') as handle:
            pickle.dump(colnames, handle, protocol=pickle.HIGHEST_PROTOCOL)

        coo_features = pd.read_csv(coo_filename,
                                   header=None,
                                   names=['row_idx', 'col_idx', 'data'],
                                   dtype={'row_idx': np.int32, 'col_idx': np.int16, 'data': np.float32},
                                   nrows=nnz)
        coo_features.sort_values(by=['row_idx', 'col_idx'], kind='mergesort', inplace=True)

        csr_indptr = self.build_indptr(coo_features, verbose=verbose)

        csr_indptr_memmap = np.memmap(os.path.join(path_to_numeric_agg, 'csr_indptr.memmap'),
                                      mode='w+',
                                      shape=csr_indptr.shape,
                                      dtype=csr_indptr.dtype)
        csr_indices_memmap = np.memmap(os.path.join(path_to_numeric_agg, 'csr_indices.memmap'),
                                       mode='w+',
                                       shape=(nnz,),
                                       dtype=coo_features.dtypes['row_idx'])
        csr_data_memmap = np.memmap(os.path.join(path_to_numeric_agg, 'csr_data.memmap'),
                                    mode='w+',
                                    shape=(nnz,),
                                    dtype=coo_features.dtypes['data'])
        csr_indptr_memmap[:] = csr_indptr[:]
        csr_indices_memmap[:] = coo_features.row_idx
        csr_data_memmap[:] = coo_features.data

        del csr_indptr
        del csr_indptr_memmap
        del csr_indices_memmap
        del csr_data_memmap

        coo_features.sort_values(by=['col_idx', 'row_idx'], kind='mergesort', inplace=True)

        csc_indptr = self.build_indptr(coo_features, compress_by='col_idx', verbose=verbose)

        csc_indptr_memmap = np.memmap(os.path.join(path_to_numeric_agg, 'csc_indptr.memmap'),
                                      mode='w+',
                                      shape=csr_indptr.shape,
                                      dtype=csr_indptr.dtype)
        csc_indices_memmap = np.memmap(os.path.join(path_to_numeric_agg, 'csc_indices.memmap'),
                                       mode='w+',
                                       shape=(nnz,),
                                       dtype=coo_features.dtypes['col_idx'])
        csc_data_memmap = np.memmap(os.path.join(path_to_numeric_agg, 'csc_data.memmap'),
                                    mode='w+',
                                    shape=(nnz,),
                                    dtype=coo_features.dtypes['data'])

        csc_indptr_memmap[:] = csc_indptr
        csc_indices_memmap[:] = coo_features.col_idx
        csc_data_memmap[:] = coo_features.data

        del csc_indptr
        del csc_indptr_memmap
        del csc_indices_memmap
        del csc_data_memmap

        sparse_matrix_info = {'shape': (self.n_operations, len(colnames)), 'nnz': nnz}
        with open(os.path.join(path_to_numeric_agg, 'sparse_matrix_info.pkl'), 'wb') as handle:
            pickle.dump(sparse_matrix_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return None




    def build_indptr(self, coo_features, compress_by='row_idx', verbose=1):
        indptr = np.zeros(self.n_operations + 1, dtype=np.int64)
        ptr = 0
        prev = -1
        chunk_size = 65536
        nnz = len(coo_features)
        n_chunks = nnz // chunk_size + (
            0 if nnz % chunk_size == 0 else 1)
        chunk_range = np.arange(chunk_size)
        for i in tqdm.tqdm(range(n_chunks), disable=(verbose == 0)):
            diffs = np.diff(coo_features[compress_by][i * chunk_size: min((i + 1) * chunk_size, nnz)].values)
            jump_points = chunk_range[1:len(diffs) + 1][diffs > 0]
            if coo_features[compress_by][i * chunk_size] > prev:
                indptr[coo_features[compress_by][i * chunk_size]] = i * chunk_size
            for jump_point in jump_points:
                indptr[coo_features[compress_by][i * chunk_size + jump_point]] = i * chunk_size + jump_point
            prev = coo_features[compress_by][min((i + 1) * chunk_size, nnz) - 1]
        indptr[-1] = nnz
        return indptr




