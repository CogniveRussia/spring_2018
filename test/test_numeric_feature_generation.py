import numpy as np
import pandas as pd

from utils.numeric_feature_aggregation_generator import PerOperationNumericFeatureAggregationGenerator

path_to_test_dataset = '/home/shared_files/alfa_test'

if __name__ == '__main__':
    feature_generator = PerOperationNumericFeatureAggregationGenerator(path_to_test_dataset)
    numeric_colname = ('all', 'P_BASEAMOUNT')
    columns_to_slice_by = [
        ('all', 'P_EKNPCODE'),
        ('all', 'P_DOCCATEGORY'),
    ]
    numeric_agg_dir = 'baseamount_agg'
    feature_generator.generate_features(numeric_colname,
                                        columns_to_slice_by,
                                        numeric_agg_dir,
                                        n_jobs=30)