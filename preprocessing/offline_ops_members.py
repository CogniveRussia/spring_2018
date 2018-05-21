import re
import os
import numpy as np
import pandas as pd

from utils import Timer
from collections import defaultdict

### PUT YOUR PATH HERE
path_to_data = '/home/shared_files/'

off_ops_filename = 'off_ops.csv'
off_members_filename = 'off_members.csv'
susp_ops_filename = 'susp_ops.csv'
susp_members_filename = 'susp_members.csv'
#susp_history_filename = ''


def read_tables_old(read_off_ops=True, read_off_members=True, read_susp_ops=True, read_susp_members=True, verbose=0):
    timer_verbose = (verbose > 0)
    result = tuple()
    if read_off_ops:
        with Timer('reading off_ops', timer_verbose):
            off_ops = pd.read_csv(os.path.join(path_to_data, off_ops_filename))
            result += (off_ops,)
    if read_off_members:
        with Timer('reading off_members', timer_verbose):
            off_members = pd.read_csv(os.path.join(path_to_data, off_members_filename))
            result += (off_members,)
    if read_susp_ops:
        with Timer('reading susp_ops', timer_verbose):
            susp_ops = pd.read_csv(os.path.join(path_to_data, susp_ops_filename))
            result += (susp_ops,)
    if read_susp_members:
        with Timer('reading susp_members', timer_verbose):
            susp_members = pd.read_csv(os.path.join(path_to_data, susp_members_filename))
            result += (susp_members,)
    return result


def read_tables(path_to_data,
                off_ops_filename=None,
                off_members_filename=None,
                susp_ops_filename=None,
                susp_members_filename=None,
                susp_history_filename=None,
                scenarios_filename=None,
                verbose=0):
    timer_verbose = (verbose > 0)
    result = tuple()
    if off_ops_filename is not None:
        with Timer('reading off_ops', timer_verbose):
            off_ops = pd.read_csv(os.path.join(path_to_data, off_ops_filename))
            result += (off_ops,)
    if off_members_filename is not None:
        with Timer('reading off_members', timer_verbose):
            off_members = pd.read_csv(os.path.join(path_to_data, off_members_filename))
            result += (off_members,)
    if susp_ops_filename is not None:
        with Timer('reading susp_ops', timer_verbose):
            susp_ops = pd.read_csv(os.path.join(path_to_data, susp_ops_filename))
            result += (susp_ops,)
    if susp_members_filename is not None:
        with Timer('reading susp_members', timer_verbose):
            susp_members = pd.read_csv(os.path.join(path_to_data, susp_members_filename))
            result += (susp_members,)
    if susp_history_filename is not None:
        with Timer('reading susp_history', timer_verbose):
            susp_history = pd.read_csv(os.path.join(path_to_data, susp_history_filename))
            result += (susp_history,)
    if scenarios_filename is not None:
        with Timer('reading scenarios', timer_verbose):
            scenarios = pd.read_csv(os.path.join(path_to_data, scenarios_filename))
            result += (scenarios,)
    return result


def read_tables(path_to_data,
                off_ops_filename=None,
                off_members_filename=None,
                susp_ops_filename=None,
                susp_members_filename=None,
                susp_history_filename=None,
                scenarios_filename=None,
                verbose=0):
    timer_verbose = (verbose > 0)
    result = tuple()
    if off_ops_filename is not None:
        with Timer('reading off_ops', timer_verbose):
            off_ops = pd.read_csv(os.path.join(path_to_data, off_ops_filename))
            result += (off_ops,)
    if off_members_filename is not None:
        with Timer('reading off_members', timer_verbose):
            off_members = pd.read_csv(os.path.join(path_to_data, off_members_filename))
            result += (off_members,)
    if susp_ops_filename is not None:
        with Timer('reading susp_ops', timer_verbose):
            susp_ops = pd.read_csv(os.path.join(path_to_data, susp_ops_filename))
            result += (susp_ops,)
    if susp_members_filename is not None:
        with Timer('reading susp_members', timer_verbose):
            susp_members = pd.read_csv(os.path.join(path_to_data, susp_members_filename))
            result += (susp_members,)
    if susp_history_filename is not None:
        with Timer('reading susp_history', timer_verbose):
            susp_history = pd.read_csv(os.path.join(path_to_data, susp_history_filename))
            result += (susp_history,)
    if scenarios_filename is not None:
        with Timer('reading scenarios', timer_verbose):
            scenarios = pd.read_csv(os.path.join(path_to_data, scenarios_filename))
            result += (scenarios,)
    return result


def process_client_indices(client_indices, trivial_ids_to_nontrivial=None, default_null=-1000):
    """
    Function that formats client ID's into integer numbers

    Parameters
    ----------
    client_indices : Series
        indices to format as integers
    trivial_to_nontrivial : dict, None
        custom duct for mapping clearly non-integer indices as '+', 'нерез', ']' to a negative numbers.
        If None passed, creates backward negative counter
    default_null : int
        integer to which null indices will be mapped

    Returns
    -------
    nontrivial_ids : int
        Indices formatted as integers
    trivial_to_nontrivial : dict
        original mapper if trivial_to_nontrivial was passed, or newly created mapping
    """
    if trivial_ids_to_nontrivial is None:
        trivial_ids_to_nontrivial = defaultdict(lambda: -len(trivial_ids_to_nontrivial) - 2)

    nontrivial_clients_ids = client_indices.copy()
    nontrivial_clients_ids[nontrivial_clients_ids.isnull()] = default_null

    nontrivial_clients_ids = nontrivial_clients_ids.map(str)

    nontrivial_ids = nontrivial_clients_ids.map(lambda s: s.strip())

    is_trivial = nontrivial_ids.map(lambda s: len(re.findall('^([\d]+|-[\d]+)', s)) == 0)
    trivial_ids = nontrivial_ids[is_trivial]

    for ti in trivial_ids:
        trivial_ids_to_nontrivial[ti]
    nontrivial_ids[is_trivial] = trivial_ids.map(lambda s: str(trivial_ids_to_nontrivial[s]))

    nontrivial_ids = nontrivial_ids.map(lambda s: re.findall('^([\d]+|-[\d]+)', s)[0]).map(int)
    return nontrivial_ids, dict(trivial_ids_to_nontrivial)


def flatten_frame_by_column(frame, to_flatten, flatten_by, group_by, names_flatten_by=None):
    """
    Function that flattens dataframe using values in specified column

    Parameters
    ----------
    frame : DataFrame
        dataframe to flatten
    to_flatten : str, list of str or DataFrame column index
        dataframe column(s) to flatten
    flatten_by : str
        column by which values frame will be flattened
    group_by: str
        column by which values flattened rows will be grouped. Note that resulted frame index will consist
        of group_by column values
    names_flatten_by: dict or None
        dict that maps flatten_by column values to level 0 column index names. If None passed,
        column values will be used instead

    Returns
    -------
    flattened_frame : DataFrame
        flattened frame

    See Also
    --------
    join_ops_with_flatten_members - join offline operations and flattened offline_members frame
    """
    columns_to_flatten = to_flatten
    column_to_flatten_by = flatten_by
    column_to_group_by = group_by
    if names_flatten_by is None:
        names_flatten_by = {
            uniq_val: uniq_val
            for uniq_val in frame[column_to_flatten_by].unique()
        }
    if not isinstance(columns_to_flatten, (list, tuple, pd.Index)):
        columns_to_flatten = [to_flatten]
    frames_to_join = [
        frame[[column_to_group_by] + columns_to_flatten][frame[column_to_flatten_by] == uniq_val]
        for uniq_val in frame[column_to_flatten_by].unique()
    ]

    for frame, uniq_val in zip(frames_to_join, frame[flatten_by].unique()):
        frame.set_index(column_to_group_by, inplace=True)
        frame.columns = pd.MultiIndex.from_product([[uniq_val], frame.columns], names=[column_to_flatten_by, 'columns'])

    flattened_frame = frames_to_join[0].join(frames_to_join[1:], how='outer')
    return flattened_frame


def join_ops_with_flatten_members(ops, flatten_ops_with_members, id_colname='ID', ops_columns_level_name=None):
    """
    Join operations and members dataframes

    Parameters
    ----------
    ops : DataFrame
        operations dataframe
    flatten_ops_with_members : DataFrame
        flattened members dataframe to join with ops
    id_colname : str
        column by which values ops frame will be joined with flatten_ops_with_members
    ops_columns_level_name:  str, None
        because flatten_ops_with_members has a multilevel index, one need to specify
        top level name for ops dataframe. by default it will equal to first top level name of
        flatten_ops_with_members

    Returns
    -------
    joined_ops : DataFrame
        joined dataframe

    See Also
    --------
    flatten_frame_by_column - Function that flattens dataframe using values in specified column
    """
    if ops_columns_level_name is None:
        ops_columns_level_name = str(list(flatten_ops_with_members.columns.levels[0]))
    ops_to_join = ops.set_index(id_colname, drop=False)
    ops_to_join.columns = pd.MultiIndex.from_product([[ops_columns_level_name], ops_to_join.columns],
                                                     names=flatten_ops_with_members.columns.names)
    joined_ops = ops_to_join.join(flatten_ops_with_members, how='left')
    joined_ops.reset_index(drop=True, inplace=True)
    return joined_ops


def off_data_cleaning(off_ops, off_members, fill_off_ops=None, fill_off_members=None, inplace=True, verbose=1):
    """
    Clean offline operations and offline members tables from broken records / fill missing values / repair broken values

    Parameters
    ----------
    off_ops : DataFrame
        offline operations dataframe
    off_members : DataFrame
        offline members dataframe
    fill_off_ops : dict, None
        default values for NaN-containing columns in offline operations
    fill_off_members : dict, None
        default values for NaN-containing columns in offline members
    inplace : bool, True
        choose to modify input tables inplace or not
    verbose:  int, 1
        verbose > 0 means that every intermediate preprocessing step will print notification
        to disable intermediate outputs, set verbose to 0

    Returns
    -------
    cleaned_off_ops : DataFrame
        cleaned off_ops dataframe
    cleaned_off_members: DataFrame
        cleaned off_members dataframe
    operationid_counter: dict
        dictionary mapping original operation indices to 0-based
        (to obtain inverse mapping compute {v: k for k, v in operationid_counter.items()})
    clientid_counter: dict
        dictionary mapping original operation indices to 0-based
        (to obtain inverse mapping compute {v: k for k, v in clientid_counter.items()})
    """
    timer_verbose = (verbose > 0)
    if not inplace:
        off_ops = off_ops.copy()
        off_members = off_members.copy()

    with Timer('drop messy off_members columns', timer_verbose):
        try:
            off_members.drop(['P_DATE_INSERT', 'P_DATE_UPDATE', 'CHANGEDATE'], axis=1, inplace=True)
        except:
            pass

    with Timer('drop off_members with OPERATIONID that are not in off_ops', timer_verbose):
        off_members = off_members[off_members.P_OPERATIONID.isin(off_ops.ID)].copy()
        off_members.reset_index(drop=True, inplace=True)

    with Timer('drop off_ops with OPERATIONID that are not in off_members', timer_verbose):
        off_ops = off_ops[off_ops.ID.isin(off_members.P_OPERATIONID)].copy()
        off_ops.reset_index(drop=True, inplace=True)

    if fill_off_members is None:
        fill_off_members = {}

    if fill_off_ops is None:
        fill_off_ops = {}

    fill_values_for_off_ops = {
        'P_EKNPCODE': fill_off_ops.get('P_EKNPCODE', -1000),
        'P_KFM_OPER_REASON': fill_off_ops.get('P_KFM_OPER_REASON', -1000)
    }

    fill_values_for_off_members = {
        'P_BSCLIENTID': fill_off_members.get('P_BSCLIENTID', -1000),
        'P_REGOPENDATE': fill_off_members.get('P_REGOPENDATE', '0000-00-00 00:00:00'),
        'P_BSACCOUNT': fill_off_members.get('P_BSACCOUNT', -1000),
        'P_BANK': fill_off_members.get('P_BANK', 'UNKNOWN'),
        'P_SDP': fill_off_members.get('P_SDP', -1000),
        'P_ORGFORM': fill_off_members.get('P_ORGFORM', -1000),
        'P_BANKCITY': fill_off_members.get('P_BANKCITY', 'UNKNOWN')
    }

    with Timer('filling NaNs in off_ops', timer_verbose):
        off_ops.fillna(value=fill_values_for_off_ops, inplace=True)

    with Timer('filling NaNs in off_members', timer_verbose):
        off_members.fillna(value=fill_values_for_off_members, inplace=True)

    with Timer('off_members client_ids processing', timer_verbose):
        off_members.loc[:, 'P_CLIENTID'], trivial_ids_to_nontrivial = process_client_indices(off_members.P_CLIENTID)

    with Timer('dropping duplicate rows from off_members', timer_verbose):
        off_members.drop_duplicates(off_members.columns.drop(['ID', 'P_BSCLIENTID', 'P_BSACCOUNT']), inplace=True)
        off_members.reset_index(drop=True, inplace=True)

    with Timer('processing off_ops', timer_verbose):
        off_ops.loc[:, 'P_OPERATIONDATETIME'] = pd.to_datetime(off_ops['P_OPERATIONDATETIME'])
        off_ops.sort_values(by='P_OPERATIONDATETIME', kind='mergesort', inplace=True)
        off_ops.reset_index(drop=True, inplace=True)

    with Timer('building counters for unique operation ID'):
        operation_id_uniqs, operation_id_indices = np.unique(off_ops.ID.values, return_index=True)
        operation_id_uniqs = operation_id_uniqs[operation_id_indices.argsort()]
        operationid_counter = {u: i for i, u in enumerate(operation_id_uniqs)}
        operationid_inv_counter = {v: k for k, v in operationid_counter.items()}


    with Timer('mapping original ID to 0-based for off_ops'):
        off_ops.loc[:, 'ID'] = off_ops['ID'].map(operationid_counter.get)

    with Timer('mapping original P_OPERATIONID to 0-based for off_members'):
        off_members.loc[:, 'P_OPERATIONID'] = off_members['P_OPERATIONID'].map(operationid_counter.get)


    with Timer('sort off-members by time-sorted operation ID', timer_verbose):
        off_members.sort_values('P_OPERATIONID', kind='mergesort', inplace=True)

    clientids_sorted = off_members['P_CLIENTID'].values

    with Timer('build fast uniqs', timer_verbose):
        clientids_uniqs, clientids_indices = np.unique(clientids_sorted[np.isfinite(clientids_sorted)], return_index=True)
        clientids_uniqs = clientids_uniqs[clientids_indices.argsort()]

    with Timer('build clientid_counter', timer_verbose):
        clientid_counter = {u: i for i, u in enumerate(clientids_uniqs)}
        clientid_inv_counter = {v: k for k, v in clientid_counter.items()}

    with Timer('mapping original P_CLIENTID to 0-based for off_members', timer_verbose):
        off_members.loc[:, 'P_CLIENTID'] = off_members['P_CLIENTID'].map(clientid_counter.get)

    with Timer('calculating seconds from start for off_ops', timer_verbose):
        off_ops['seconds_from_start'] = (off_ops.P_OPERATIONDATETIME -
                                         off_ops.P_OPERATIONDATETIME.min()).dt.total_seconds()

    off_members.reset_index(drop=True, inplace=True)
    off_ops.reset_index(drop=True, inplace=True)

    with Timer('joining off_members and seconds_from_start from off_ops', timer_verbose):
        off_members = off_members.merge(off_ops[['ID', 'seconds_from_start']], how='inner', left_on='P_OPERATIONID',
                                        right_on='ID')
        off_members.drop(['ID_y'], axis=1, inplace=True)
        off_members.rename(columns={'ID_x': 'ID'}, inplace=True)

    with Timer('removing duplicated P_OPERATIONID-P_CLIENTID-P_CLIENTROLE rows, preserving last ones'):
        duplicate_row_detector = off_members.groupby(['P_OPERATIONID', 'P_CLIENTID', 'P_CLIENTROLE']).size()

        duplicate_rows = duplicate_row_detector[duplicate_row_detector > 1]
        if len(duplicate_rows) != 0:
            duplicate_rows = duplicate_rows.reset_index(drop=False)
            duplicate_indices_to_delete = pd.Index([])
            for rownum, row in duplicate_rows.iterrows():
                duplicate_indices_to_delete = duplicate_indices_to_delete.append(off_members[(off_members.P_OPERATIONID == row.P_OPERATIONID)
                                              & (off_members.P_CLIENTID == row.P_CLIENTID)
                                              & (off_members.P_CLIENTROLE == row.P_CLIENTROLE)].index[:-1])
            off_members.drop(duplicate_indices_to_delete, inplace=True)
            del duplicate_indices_to_delete
        del duplicate_row_detector
        del duplicate_rows

    with Timer('removing duplicated P_OPERATIONID-P_CLIENTROLE rows, preserving first ones'):
        duplicate_row_detector = cleaned_off_members.groupby(['P_OPERATIONID', 'P_CLIENTROLE']).size()

        duplicate_rows = duplicate_row_detector[duplicate_row_detector > 1]
        if len(duplicate_rows) != 0:
            duplicate_rows = duplicate_rows.reset_index(drop=False)
            duplicate_indices_to_delete = pd.Index([])
            for rownum, row in duplicate_rows.iterrows():
                duplicate_indices_to_delete = duplicate_indices_to_delete.append(
                    cleaned_off_members[(cleaned_off_members.P_OPERATIONID == row.P_OPERATIONID)
                                        & (cleaned_off_members.P_CLIENTROLE == row.P_CLIENTROLE)].index[1:])
            cleaned_off_members.drop(duplicate_indices_to_delete, inplace=True)
            del duplicate_indices_to_delete
        del duplicate_row_detector
        del duplicate_rows

    with Timer('stable sorting off_members by P_CLIENTID inplace', timer_verbose):
        off_members.sort_values('P_CLIENTID', kind='mergesort', inplace=True)

    with Timer('retrieve deltas between client current and last operation', timer_verbose):
        off_members['seconds_from_last_client_op'] = off_members.groupby('P_CLIENTID')['seconds_from_start'] \
            .agg('diff') \
            .sort_index(kind='mergesort') \
            .fillna(-100000)

    with Timer('sort back off_members by P_OPERATIONID', timer_verbose):
        off_members.sort_values('P_OPERATIONID', kind='mergesort', inplace=True)

    with Timer('retrieving member operationdatetime for each row from off_members', timer_verbose):
        #member_operationdatetime = off_ops.P_OPERATIONDATETIME[off_members.P_OPERATIONID]
        member_operationdatetime = pd.to_datetime(
            pd.Series(
                np.repeat(off_ops.P_OPERATIONDATETIME.values, off_members.groupby('P_OPERATIONID').size().values)
            )
        )

    with Timer('computing acc_persistence for off_members', timer_verbose):
        regopendate = pd.to_datetime(off_members['P_REGOPENDATE'],errors='coerce', format='%Y-%m-%d %H:%M:%S')
        acc_persistence = (member_operationdatetime.reset_index(drop=True) - regopendate.reset_index(drop=True)).dt.days.copy()
        acc_persistence.loc[acc_persistence.isnull()] = -100000
        off_members['acc_persistence'] = acc_persistence.values
        off_members.reset_index(drop=True, inplace=True)

    with Timer('transforming P_EKNPCODES into proper ints', timer_verbose):
        off_ops.loc[:, 'P_EKNPCODE'] = off_ops['P_EKNPCODE'].map(lambda s: int(float(str(s).replace('З', '3'))))

    return off_ops, off_members, operationid_counter, clientid_counter


if __name__ == '__main__':
    off_ops, off_members, susp_ops, susp_members = read_tables(path_to_data, off_ops_filename, off_members_filename, susp_ops_filename, susp_members_filename, verbose=1)

    with Timer('cleaning off_ops and off_members', verbose=1):
        cleaned_off_ops, cleaned_off_members, operationid_counter, clientid_counter = off_data_cleaning(off_ops,
                                                                                                        off_members,
                                                                                                        verbose=1)