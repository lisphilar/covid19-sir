# Dataset for analysing COVID-19 outbreak

In this directory, datasets for analysing COVID-19 outbreak are archived.

# COVID-19 dataset in Japan

## 1. Context

This is a COVID-19 dataset in Japan.  This does not include the cases in Diamond Princess cruise ship (Yokohama city, Kanagawa prefecture) and Costa Atlantica cruise ship (Nagasaki city, Nagasaki prefecture).

- Total number of cases in Japan
- The number of vaccinated people (New/experimental)
- The number of cases at prefecture level
- Metadata of each prefecture

This dataset can be retrieved with CovsirPhy (Python library).

```bash
pip install covsirphy --upgrade
```

```Python
import covsirphy as cs
data_loader = cs.DataLoader()
japan_data = data_loader.japan()
# The number of cases (Total/each province)
clean_df = japan_data.cleaned()
# Metadata
meta_df = japan_data.meta()
```

Please refer to [CovsirPhy Documentation: Japan-specific dataset](https://lisphilar.github.io/covid19-sir/usage_dataset.html#Japan-specific-dataset).


Before analysing the data, please refer to [Kaggle notebook: EDA of Japan dataset](https://www.kaggle.com/lisphilar/eda-of-japan-dataset) and [COVID-19: Government/JHU data in Japan](https://www.kaggle.com/lisphilar/covid-19-government-jhu-data-in-japan).
The detailed explanation of the build process is discussed in [Steps to build the dataset in Japan](https://www.kaggle.com/lisphilar/covid19-dataset-in-japan/discussion/148766).

### 1.1 Total number of cases in Japan

`covid_jpn_total.csv`  
Cumulative number of cases:

- PCR-tested / PCR-tested and positive
- with symptoms (to 08May2020) / without symptoms (to 08May2020) / unknown (to 08May2020) 
- discharged
- fatal

The number of cases:

- requiring hospitalization (from 09May2020)
- hospitalized with mild symptoms (to 08May2020)  / severe symptoms / unknown (to 08May2020) 
- requiring hospitalization, but waiting in hotels or at home (to 08May2020)

In primary source, some variables were removed on 09May2020. Values are NA in this dataset from 09May2020.

Manually collected the data from Ministry of Health, Labour and Welfare HP:  
[厚生労働省 HP (in Japanese)](https://www.mhlw.go.jp/)  
[Ministry of Health, Labour and Welfare HP (in English)](https://www.mhlw.go.jp/english/)

The number of vaccinated people:

- `Vaccinated_1st`: the number of vaccinated persons for the first time on the date
- `Vaccinated_2nd`: the number of vaccinated persons with the second dose on the date

Data sources for vaccination:

- [厚生労働省 HP 新型コロナワクチンの接種実績(in Japanese)](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/vaccine_sesshujisseki.html)
- [首相官邸 新型コロナワクチンについて](https://www.kantei.go.jp/jp/headline/kansensho/vaccine.html)
- [Twitter: 首相官邸（新型コロナワクチン情報）](https://twitter.com/kantei_vaccine)

### 1.2 The number of cases at prefecture level

`covid_jpn_prefecture.csv`  
Cumulative number of cases:

- PCR-tested / PCR-tested and positive
- discharged
- fatal

The number of cases:

- requiring hospitalization (from 09May2020)
- hospitalized with severe symptoms (from 09May2020)

Using pdf-excel converter, manually collected the data from Ministry of Health, Labour and Welfare HP:  
[厚生労働省 HP (in Japanese)](https://www.mhlw.go.jp/)  
[Ministry of Health, Labour and Welfare HP (in English)](https://www.mhlw.go.jp/english/)

Note:
`covid_jpn_prefecture.groupby("Date").sum()` does not match `covid_jpn_total`.
When you analyse total data in Japan, please use `covid_jpn_total` data.

### 1.3 Metadata of each prefecture

`covid_jpn_metadata.csv`  

- Population (Total, Male, Female): [厚生労働省 厚生統計要覧（2017年度）第１－５表](https://www.mhlw.go.jp/toukei/youran/indexyk_1_1.html) 
- Area (Total, Habitable): [Wikipedia 都道府県の面積一覧 (2015)](https://ja.wikipedia.org/wiki/%E9%83%BD%E9%81%93%E5%BA%9C%E7%9C%8C%E3%81%AE%E9%9D%A2%E7%A9%8D%E4%B8%80%E8%A6%A7#cite_note-2)

- Hospital_bed:
With the primary data of [厚生労働省 感染症指定医療機関の指定状況（平成31年4月1日現在）](https://www.mhlw.go.jp/bunya/kenkou/kekkaku-kansenshou15/02-02.html), [厚生労働省 第二種感染症指定医療機関の指定状況（平成31年4月1日現在）](https://www.mhlw.go.jp/bunya/kenkou/kekkaku-kansenshou15/02-02-01.html), [厚生労働省 医療施設動態調査（令和２年１月末概数）](https://www.mhlw.go.jp/toukei/saikin/hw/iryosd/m20/is2001.html), [厚生労働省 感染症指定医療機関について](https://www.mhlw.go.jp/bunya/kenkou/kekkaku-kansenshou19/dl/20140811_01.pdf) and secondary data of [COVID-19 Japan 都道府県別 感染症病床数](https://code4sabae.github.io/bedforinfection/),

    - Specific: Hospital beds of medical institutions designated for specific infectious diseases
    - Type-I: Hospital beds of medical institutions designated for type I infectious diseases
    - Type-II: Hospital beds of medical institutions designated for type II infectious diseases
    - Tuberculosis: Hospital beds of medical institutions designated for tuberculosis (outpatient care)
    - Care: long term care bed of hospitals
    - Total: Beds of all hospitals

- Clinic_bed:
With the primary data of [医療施設動態調査（令和２年１月末概数）](https://www.mhlw.go.jp/toukei/saikin/hw/iryosd/m20/is2001.html) ,

    - Care: long term care beds of clinics
    - Total: Beds of all clinics

- Location: Data is from  [LinkData 都道府県庁所在地 (Public Domain)](http://linkdata.org/work/rdf1s8i) (secondary data).

    - Latitude
    - Longitude

- Admin

    - Capital: Prefectural capital city. Data is from  [LinkData 都道府県庁所在地 (Public Domain)](http://linkdata.org/work/rdf1s8i) (secondary data).
    - Region: Region name. Data is from [WIkipedia ](https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E3%81%AE%E5%9C%B0%E5%9F%37F) (secondary data). "Kyushu-Okinawa region" was separated to "Kyushu" and "Okinawa" by this datasets' author.
    - Num: Prefecture code (JIS X 0401: Hokkaido=1,...Okinawa=47). Data is from [国土交通省 GIS HP Pref code](http://nlftp.mlit.go.jp/ksj/gml/codelist/PrefCd.html). cf. (not source) [Japan VIsitor: Japan Prefectures Map](https://www.japanvisitor.com/japan-travel/prefectures-map).

## 2. Acknowledgements

To create this dataset,  edited and transformed data of the following sites was used.

厚生労働省 Ministry of Health, Labour and Welfare, Japan:  
[厚生労働省 HP (in Japanese)](https://www.mhlw.go.jp/)  
[Ministry of Health, Labour and Welfare HP (in English)](https://www.mhlw.go.jp/english/)
[厚生労働省 HP 利用規約・リンク・著作権等 CC BY  4.0 (in Japanese)](https://www.mhlw.go.jp/chosakuken/index.html)

国土交通省 Ministry of Land, Infrastructure, Transport and Tourism, Japan:
[国土交通省 HP (in Japanese)](http://www.mlit.go.jp/)
[国土交通省 HP (in English)](http://www.mlit.go.jp/en/)
[国土交通省 HP 利用規約・リンク・著作権等 CC BY  4.0 (in Japanese)](http://www.mlit.go.jp/link.html)

Code for Japan / COVID-19 Japan:
[Code for Japan](https://www.code4japan.org/)
[COVID-19 Japan Dashboard (CC BY 4.0)](https://www.stopcovid19.jp/)
[COVID-19 Japan 都道府県別 感染症病床数 (CC BY)](https://code4sabae.github.io/bedforinfection/)

Wikipedia:
[Wikipedia](https://ja.wikipedia.org/wiki/)

LinkData:
[LinkData (Public Domain)](http://linkdata.org/)

## 3. Inspiration

1. Changes in number of cases over time
2. Percentage of patients without symptoms / mild or severe symptoms
3. What to do next to prevent outbreak

## License and how to cite

Kindly cite this dataset under CC BY-4.0 license as follows.
- Hirokazu Takaya (2020-2021), COVID-19 dataset in Japan, GitHub repository, https://github.com/lisphilar/covid19-sir/data/japan, or
- Hirokazu Takaya (2020-2021), COVID-19 dataset in Japan, Kaggle Dataset, https://www.kaggle.com/lisphilar/covid19-dataset-in-japan
