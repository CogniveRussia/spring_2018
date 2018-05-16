from enum import Enum
from tqdm import tqdm_notebook

import cx_Oracle
import os
import pandas as pd

class Tables(Enum):
    OFF_OPS = 1
    OFF_MEMS = 2
    SUSP_OPS = 3
    SUSP_MEMS = 4

class DBManager:
    TABLE_NAMES = { Tables.OFF_OPS: "TB_OFFLINEOPERATIONS",
                    Tables.OFF_MEMS: "MV_OFF_MEMBERS",
                    Tables.SUSP_OPS: "TB_SUSPICIOUSOPERATIONS",
                    Tables.SUSP_MEMS: "MV_SUSP_MEMBERS"}
    
    COLUMNS_TO_EXCLUDE = ["P_HISTORY", "P_ORDERNUMBER", "P_DOCNUMBER", "P_USERNAME", "P_PROPERTY", 
                          "P_PROPERTYNUMBER", "ABIS_PRODUCT", "P_SDP", "RISK_CHECKED", "ABIS_SERVICE"]
    
    def __init__(self, oracle_user, oracle_pass, oracle_scheme, oracle_host):
        self.scheme = oracle_scheme
        self.connection = cx_Oracle.connect(oracle_user, oracle_pass, oracle_host, encoding = "UTF-8", nencoding = "UTF-8")
        self.cursor = self.connection.cursor()
        
    def get_table_name(self, table):
        return "{0}.{1}".format(self.scheme, DBManager.TABLE_NAMES[table])
    
    def get_table_length(self, table):
        table_name = self.get_table_name(table)
        
        query = """SELECT COUNT(*) FROM {0}""".format(table_name)
        return next(self.cursor.execute(query))[0]
    
    def read_table(self, table, row_limit=None):
        table_name = self.get_table_name(table)
        
        row_limit_expression = ""
        if row_limit is not None:
            row_limit_expression = "WHERE ROWNUM < {limit}".format(limit=row_limit)
            
        query = "SELECT * FROM {table} {row_limit_expr}".format(table=table_name, row_limit_expr=row_limit_expression)
        return pd.read_sql(query, self.connection)
    
    def get_columns(self, table):
        query = """SELECT * FROM {0} WHERE ROWNUM < 2""".format(self.get_table_name(table))
        return pd.read_sql(query, self.connection).columns
    
    def dump_table(self, table, path, columns=None, batch_size=10**5):
        table_with_schema = self.get_table_name(table)
        row_count = self.get_table_length(table)
        
        if columns is None:
            columns = self.get_columns(table)

        query = "SELECT {cols} FROM {table}".format(cols=",".join(columns), table=table_with_schema)
        self.cursor.execute(query)
        
        table_name = str(table).split(".")[1]

        position = 0
        csv_created = False

        with tqdm_notebook(total=row_count, desc=table_name) as pbar:
            while position < row_count:
                rows_to_fetch = min(batch_size, row_count-position)

                data = self.cursor.fetchmany(rows_to_fetch)
                dataframe = pd.DataFrame(data=data, columns=columns)

                if not csv_created:
                    dataframe.to_csv(path, index=False)
                    csv_created = True
                else:
                    dataframe.to_csv(path, mode="a", index=False, header=False)

                position += batch_size
                pbar.update(rows_to_fetch)
                
    def _exclude_columns(self, columns):
        return list(set(columns) - set(DBManager.COLUMNS_TO_EXCLUDE))
    
    def dump_all_tables(self, folder):
        for table in DBManager.TABLE_NAMES:
            table_with_schema = self.get_table_name(table)
            table_name = str(table).split(".")[1].lower()
            table_dump_path = os.path.join(folder, table_name + ".csv")
            
            columns = self.get_columns(table)
            filtered_columns = self._exclude_columns(columns)
            
            self.dump_table(table, table_dump_path, columns=filtered_columns)