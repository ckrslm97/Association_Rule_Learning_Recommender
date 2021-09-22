###############################################

# ASSOCIATION RULE LEARNING RECOMMENDER #

###############################################

# !pip install mlxtend
import pandas as pd

## GÖREV - 1 ##

# Veri Ön İşleme

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("C:\\Users\\ckrsl\\Desktop\\VERİ BİLİMİ\\PROJELER\\pythonProject2\\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# Veriyi anlamak için gerekenleri yazdırır.
def dataframe_info(df):

    print("-----Head-----","\n",df.head())

    print("\n-----Tail-----","\n",df.tail())

    print("\n-----Shape-----","\n",df.shape)

    print("\n-----Columns-----","\n",df.columns)

    print("\n-----Index-----","\n",df.index)

    print("\n-----Statistical Values-----","\n",df.describe().T)

dataframe_info(df)

### Aykırı değerlerden kurtulma ###
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)

## GÖREV - 2 ##

##### ARL Veri Yapısını Hazırlama (Invoice - Product Matrix) #####

# Germany Müşterileri'nin Bilgilerini başka bir dataframe'e aktarma

df_ger = df[df['Country'] == "Germany"]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


gr_inv_pro_df = create_invoice_product_df(df_ger,id =True)

## ARL Veri Yapısı ##
print("---------------ARL Veri Yapısı----------------\n")
print(gr_inv_pro_df.head(10))


## Birliktelik Kurallarının Çıkartılması ##

# Tüm olası ürün birlikteliklerinin olasılıkları
frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)


# Birliktelik kurallarının çıkarılması:
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head(500)

# support: İkisinin birlikte görülme olasılığı
# confidence: X alındığında Y alınma olasılığı.
# lift: X alındığında Y alınma olasılığı şu kadar kat artıyor.

## GÖREV - 3 ##

# Id'leri verilen ürünlerin isimlerini yazdırma

def check_id(dataframe,stock_code):
    product_name  = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(f"{stock_code} ID'li ürünün adı : {product_name}")

check_id(df,21987)
print("\n")
check_id(df,23235)
print("\n")
check_id(df,22747)
print("\n")

## GÖREV - 4 ##

# Sepetteki kullanıcılar için öneri yapma #

sorted_rules = rules.sort_values("lift", ascending=False)


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.loc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

print("\n")

print("-------- ÜRÜN ÖNERİLERİ -----------\n")

print("21987 id'li müşteri için ürün önerisi: ",arl_recommender(rules, 21987, 1))
print(check_id(df,21988))
print("\n")

print("23235 id'li müşteri için ürün önerisi : ",arl_recommender(rules, 23235, 1))
print(check_id(df,23236))
print("\n")

print("22747 id'li müşteri için ürün önerisi : ",arl_recommender(rules, 22747, 1))
print(check_id(df,22745))

