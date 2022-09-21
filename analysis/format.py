def shorten_model_name(mp):
    if 'multilingual' in mp:
        short_name = 'mBert'
    elif mp.startswith('bert') or mp.startswith('rubert'):
        short_name = 'Bert'
    elif mp.startswith('albert'):
        short_name = 'Albert'
    elif mp.startswith('distilbert'):
        short_name = 'DistilBert'
    elif mp == 'human':
        short_name = 'Human'
    else:
        print(mp)
        raise ValueError
    return short_name

def shorten_importance_type(it):
    return {
        '-': '-',
        'attention': 'Attn (last)',
        'saliency': 'Saliency',
        'flow': 'Flow',
        'attention_1st_layer': 'Attn (1st)'}[it]

def format_corpus_name(c):
    return {
        'geco_nl': 'Dutch',
        'geco': 'English (Geco)',
        'potsdam': 'German',
        'russsent': 'Russian',
        'zuco':  'English (ZuCo)'
    }[c]

def format_df(df):
    df['model'] = df['model'].apply(shorten_model_name)
    df['importance_type'] = df['importance_type'].apply(shorten_importance_type)
    df['corpus'] = df['corpus'].apply(format_corpus_name)
    df = df.sort_values(['corpus', 'model', 'importance_type'])
    return df
