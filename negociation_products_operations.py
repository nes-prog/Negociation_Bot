import pandas as pd 
from math import log10

# upload produdts.csv
products = pd.read_csv('products.csv')
quantity = 100
# calculate quantity discount
quantity_discount = 2.5*log10(quantity)
#calculate price before negociation and add a column price_befor_negotiation
products["price_before_neg"] = products["cost_price"]+(products["cost_price"]*products["revenue"])/100

#calculate sd
def get_level(x):
  '''
    get level 
  '''
  if x>=1000 and x<=5000:
    level = 1 
  elif  x>5000 and x<=10000:
    level = 2
  elif  x>=15000 and x<30000:
    level = 3
  elif  x>=30000 and x<60000:
    level = 4
  else:
    level = 5
  return level 
products["sales_discount"] = products["prev_sales"].apply(lambda x: get_level(x)*2)
products

def calculate_last_price(product_id, quantity, user_id):
  '''
  get the last price of a given product_id, quantity, user_id
  
  '''
  products["sales_discount"] = products["prev_sales"].apply(lambda x: get_level(x)*2)
  rows_user_id = products.loc[products['user_id'] == user_id ]
  rows_retrieve_prod = rows_user_id.loc[rows_user_id['uniq_id']==product_id]
  price = rows_retrieve_prod["price_before_neg"]
  last_price = rows_retrieve_prod["cost_price"]*(1+(rows_retrieve_prod["revenue"]-quantity_discount-rows_retrieve_prod["sales_discount"])/100)
  return last_price, price, rows_retrieve_prod.index.to_list()
