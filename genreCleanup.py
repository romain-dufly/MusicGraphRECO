
import pandas as pd

## Load dataframe with genre tags

df = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/genres.csv', sep=';')

## Clean up genre tags

for i in range(len(df)):
    print(i)
    if 'filmi' in df.iloc[i,3] or 'bollywood' in df.iloc[i,3] :
        df.iloc[i,3] = 'filmi'
    elif 'mexicana' in df.iloc[i,3] :
        df.iloc[i,3] = 'mexicana'
    elif 'sertanejo' in df.iloc[i,3] :
        df.iloc[i,3] = 'brazilian'
    elif 'reggaeton' in df.iloc[i,3] or 'samba' in df.iloc[i,3] or 'salsa' in df.iloc[i,3] or 'bachata' in df.iloc[i,3]:
        df.iloc[i,3] = 'baile latino'
    elif 'classical' in df.iloc[i,3] :
        df.iloc[i,3] = 'classical'
    elif 'country' in df.iloc[i,3] :
        df.iloc[i,3] = 'country'
    elif 'metal' in df.iloc[i,3] :
        df.iloc[i,3] = 'metal'
    elif 'reggae' in df.iloc[i,3] :
        df.iloc[i,3] = 'reggae'
    elif 'rap' in df.iloc[i,3] :
        df.iloc[i,3] = 'rap'
    elif 'hip hop' in df.iloc[i,3] :
        df.iloc[i,3] = 'hip-hop'
    elif 'edm' in df.iloc[i,3] :
        df.iloc[i,3] = 'edm'
    elif 'rock' in df.iloc[i,3] :
        df.iloc[i,3] = 'rock'
    elif 'r&b' in df.iloc[i,3] :
        df.iloc[i,3] = 'r&b'
    elif 'jazz' in df.iloc[i,3] :
        df.iloc[i,3] = 'jazz'
    elif 'soul' in df.iloc[i,3] :
        df.iloc[i,3] = 'soul'
    elif 'pop' in df.iloc[i,3] :
        df.iloc[i,3] = 'pop'
    elif 'movie' in df.iloc[i,3] or 'hollywood' in df.iloc[i,3] :
        df.iloc[i,3] = 'movie'
    else:
        df.iloc[i,3] = 'other'

df.to_csv('/Users/romaindufly/Desktop/Projet MODAL/genresClean.csv', sep=';')