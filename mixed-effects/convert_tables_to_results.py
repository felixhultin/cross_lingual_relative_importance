import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None



def convert_tables(table4, table6, with_reffect: bool = True):
    LMM = 'LMMwith_reffect' if with_reffect else 'LMMwithout_reffect'
    table6 = table6[table6['LMM'] == LMM]
    rows = []
    for group, values in table6.groupby(['corpus', 'importance_type', 'model']):
        corpus, importance_type, model = group
        row = {'corpus': corpus, 'model': model, 'importance_type': importance_type}
        for idx, v in values.iterrows():
            row[v['indep_vars']] = v['R2m']
        rows.append(row)


    df = pd.DataFrame.from_records(rows)
    for c in table4['model'].unique():
        c_values = []
        for idx, row in df.iterrows():
            r2m = table4[\
                (table4['corpus'] == row['corpus']) &\
                (table4['model'] == c) &\
                (table4['LMM'] == LMM)\
            ]['R2m'].iloc[0]
            c_values.append(r2m)
        df[c] = c_values


    mapper = {
        'freq': 'human~freq',
        'length': 'human~length',
        'both': 'human~freq+length',
        'log_lm_importance': 'human~model',
        'log_lm_importance log_freq': 'human~model+freq',
        'log_lm_importance log_length': 'human~model+length',
        'log_lm_importance log_freq log_length': 'human~model+freq+length'
    }

    return df.rename(columns=mapper)

if __name__ == '__main__':
    table4 = pd.read_csv('Table4.csv', delimiter='\t')
    table6 = pd.read_csv('Table6.csv', delimiter='\t')

    df_with_reffect = convert_tables(table4, table6, with_reffect = True)
    df_without_reffect = convert_tables(table4, table6, with_reffect = False)

    with pd.ExcelWriter('conversion.xlsx') as writer:
        df_with_reffect.to_excel(writer, sheet_name='with_reffect', index=False)
        df_without_reffect.to_excel(writer, sheet_name='without_reffect', index=False)
