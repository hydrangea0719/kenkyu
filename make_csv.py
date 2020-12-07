# make_csv.py
# csvファイルを作る

import pandas as pd

import dataset.original.mainichi_dataset as maidata


def main():
    for genre in maidata.KijiGenres.keys():
        print(f"{genre}                      ")
        for year in range(2008, 2013):
            print(f'\r{year}                                  ')
            kijis = maidata.file_to_kijis(year, f"dataset/original/mai{year}.txt",
                                          [lambda kiji: kiji.genre == genre])
            df: pd.DataFrame = maidata.to_pd_dataframe(kijis)
            df_len = len(df) - 1
            df['honbun'] = ['[SEP]'.join(honbuns) for i, honbuns in enumerate(df['honbuns'])
                            if print(f"\rhonbun remake {i}/{df_len}          ", end='') or True]
            df = df[['midashi', 'honbun', 'genre', 'has_photo', 'day_page', 'search_number', 'day_time_ect', 'has_cho']]
            df.to_csv(f"dataset/original_csv/mai{year}_{genre}.csv", encoding='utf-8')

        df = pd.concat(
            [pd.read_csv(f"dataset/original_csv/mai{year}_{genre}.csv", encoding='utf-8', index_col=0)
             for year in range(2008, 2013)])
        df.to_csv(f"dataset/original_csv/mai(2008_2013)_{genre}.csv", encoding='utf-8')
        print("\r                              ")


def make_123():  # after main
    df = pd.concat(
        [pd.read_csv(f"dataset/original_csv/mai(2008_2013)_{genre}.csv", encoding='utf-8', index_col=0)
         for genre in ['０１', '０２', '０３']])
    df.to_csv(f"dataset/original_csv/mai(2008_2013)_(０１,０２,０３).csv", encoding='utf-8')


if __name__ == '__main__':
    main()
    make_123()
