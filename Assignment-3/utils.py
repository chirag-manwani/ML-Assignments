def binarize_median(
    df,
    columns
):
    '''
    Input- 
        df- dataframe
        columns- columns to binarize
    Output-
        p_df- processed dataframe
    '''
    for col in columns:
        median = df[col].median()
        df[col] = (df[col] >= median).astype(int)
    return df