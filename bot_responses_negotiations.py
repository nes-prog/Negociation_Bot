from tkinter import *
from model_bert import *
from negociation_products_operations import * 
import pickle
import spacy
from word2number import w2n

NER = spacy.load("en_core_web_sm")

dictio = {}
for intent in intents['intents']:

        #add documents in the corpus
        key = intent["tag"]
        dictio.update({key : intent["responses"]})
dictio
# with open('parrot.pkl', 'wb') as f:
#     pickle.dump(data.pkl, f)
list_user_product = []
user_price = []
file_ = "user_price.pkl"
file = "cmp.pkl"
file_name = "data.pkl"
file_n = "number_neg.pkl"
cpm_neg = [0]
cmp = [0]
user_price = []
#pickle file for the list_user_product, in this list we will add(user_id, product_id, quantity, price, last price)
open_file = open(file_name, "wb")
pickle.dump(list_user_product, open_file)
open_file.close()
#pickle file for the number of negociation
open_file = open(file_n, "wb")
pickle.dump(cpm_neg, open_file)
open_file.close()
#pickle file for the price suggested by the user
open_file = open(file_, "wb")
pickle.dump(user_price, open_file)
open_file.close()
#pickle file a variable to distinguish if it is the quantitity or the price
open_file = open(file, "wb")
pickle.dump(cmp, open_file)
open_file.close()

def calculate_negociation(cmp_neg ):
            with open("user_price.pkl",'rb') as wfp:
                user_price= pickle.load(wfp)
            with open("data.pkl",'rb') as wfp:
                list_user_product= pickle.load(wfp)
            if len(list_user_product) == 6 :
                price = float(list_user_product[3])
            else : 
                price = float(list_user_product[len(list_user_product)-1])              
            last_price = float(list_user_product[4]) 
            indexe = int(list_user_product[5])
            result = ""
            # the owner of the products set a specific number of negociation (2 for example)
            # if cmp_neg[0]== 2:
            #     result = "cant'get below "+str(round(list_user_product[len(list_user_product)-1], 2))
            #     return result, indexe
            # and(cmp_neg[0] <2)  must be added to the condition if (price >= last_price )
            if (price >= last_price ) :
                    med_price = price - (price - last_price)/2
                    if price > med_price:
                        price = price - (price *2.5)/100
                    else : 
                        price = price -(price *5)/100
                        
                    if price < float(user_price[0]):
                        price = float(user_price[0])
                        result = "i propose the same price "+ str(price)+" that you've suggested at the begining"
                        return result, indexe
                    if (price <= last_price) :
                       print("ok")
                       result = "cant'get below "+str(round(last_price, 2))
                       list_user_product.append(last_price)
                       with open("data.pkl",'wb') as wfp:
                            pickle.dump(list_user_product, wfp)
                    else: 
                            cmp_neg[0]+=1
                            with open("number_neg.pkl",'wb') as wfp:
                                pickle.dump(cmp_neg, wfp)
                            result = "what about " +str(round(price, 2))
                            list_user_product.append(price)
                            with open("data.pkl",'wb') as wfp:
                                pickle.dump(list_user_product, wfp)
                    return result, indexe

def get_response(user_input, max_seq_len):
    
    with open("data.pkl",'rb') as rfp: 
        list_user_product= pickle.load(rfp)
  
    with open("number_neg.pkl",'rb') as rfp: 
        cmp_neg= pickle.load(rfp)
    text1= NER(user_input) 
    for word in text1.ents:
        if (len(list_user_product) == 2) and (word.label_ == "CARDINAL"):
            cmp[0] += 1
            with open("cmp.pkl",'wb') as wfp:
                pickle.dump(cmp, wfp)
            quant = float(w2n.word_to_num(word.text))
            list_user_product.append(quant)
            # print(list_user_product)
            last_price, price, indexe =calculate_last_price(list_user_product[1], float(quant), list_user_product[0])
            price = price.iloc[0]
            last_price = last_price.iloc[0]
            list_user_product.append(price)
            list_user_product.append(last_price)
            list_user_product.append(indexe[0])
            print(list_user_product)
            with open("data.pkl",'wb') as wfp: 
                pickle.dump(list_user_product, wfp)
            message = "which price you propose"
            return message

            
      
    if user_input.isdigit() == True:
        cmp[0] += 1
        with open("cmp.pkl",'wb') as wfp:
            pickle.dump(cmp, wfp)
        if cmp[0] == 1:
        # intent_predicted =="id"
            list_user_product.append(user_input)
            last_price, price, indexe = calculate_last_price(list_user_product[1], float(user_input), list_user_product[0])
            price = price.iloc[0]
            last_price = last_price.iloc[0]
            message =  "which price you propose"
            list_user_product.append(price)
            list_user_product.append(last_price)
            list_user_product.append(indexe[0])
            print(list_user_product)
            with open("data.pkl",'wb') as wfp: 
                pickle.dump(list_user_product, wfp)
        else :
            user_price.append(user_input)
            with open("user_price.pkl",'wb') as wfp: 
                pickle.dump(user_price, wfp)
            
            if float(user_input)> list_user_product[3]:
                message = "okey that's good for both of us"
                indexe = int(list_user_product[5])
                products.at[indexe,"price_conversation"]=user_input
                print(products)
                products.to_csv("products.csv", index = False)
            else : 
                print("pre", cmp)
                print("proposed", user_input)
                print("price", list_user_product[3])
                message = "No but the product price is "+ str(round(list_user_product[3]))
                
    # last_price, price = calculate_last_price("eac7efa5dbd3d667f26eb3d3ab504464", 1, "c2d766ca982eca8304150849735ffef9")   
    elif (len(user_input) >8) and  (" " not in (user_input)) and (len(list_user_product)<=2) :
        # intent_predicted =="id"
        list_user_product.append(user_input) 
        
        with open("data.pkl",'wb') as wfp:
            pickle.dump(list_user_product, wfp)
            
        with open("data.pkl",'rb') as wfp:
            list_user_product= pickle.load(wfp)
            
        if len(list_user_product) == 2:
            message = " How many products do you need"

        else : 
            message = "give me the id of the product that you are looking for " 
    else :
            
        print(user_input) 
        intent_predicted = get_prediction(user_input, max_seq_len)
        if (intent_predicted == "buy") or (intent_predicted == "welcome") or (intent_predicted == "greeting") or (intent_predicted == "goodbye") or (intent_predicted == "thank"):
            message = dictio[intent_predicted][0]
            
        elif (intent_predicted == "id_user" ):
            list_user_product.append(user_input.split("is ")[1])
            with open("data.pkl",'wb') as wfp:
                pickle.dump(list_user_product, wfp)
            with open("data.pkl",'rb') as wfp:
                list_user_product= pickle.load(wfp)
            message = dictio[intent_predicted][0]
            
        elif (intent_predicted == "quantity" ):
            cmp[0] += 1
            with open("cmp.pkl",'wb') as wfp:
                pickle.dump(cmp, wfp)
            if cmp[0] == 1:
                i = 0
                quant = ""
                tokens = user_input.split(" ")
                while i < len(tokens):
                    if tokens[i].isdigit() == True:
                        quant = float(tokens[i].isdigit())
                    else:
                        i+=1
                if (i == len(tokens)) and (len(quant) == 0):
                    text1= NER(user_input)
                    for word in text1.ents:
                        # print(word.text,word.label_)
                        if word.label_ == "CARDINAL":
                            quant = float(w2n.word_to_num(word.text))
                            print(type(quant))
                list_user_product.append(quant)
                print(list_user_product)
                last_price, price, indexe =calculate_last_price(list_user_product[1], float(quant), list_user_product[0])
                price = price.iloc[0]
                last_price = last_price.iloc[0]
                message =  "which price you propose"
                list_user_product.append(price)
                list_user_product.append(last_price)
                list_user_product.append(indexe[0])
                print(list_user_product)
                with open("data.pkl",'wb') as wfp: 
                    pickle.dump(list_user_product, wfp)
            

        elif intent_predicted == "id_product": 
            with open("data.pkl",'rb') as rfp: 
                  list_user_product= pickle.load(rfp)
            product_id = user_input.split("is ")[1]
            list_user_product.append(product_id)
            message ="How many products you need"

            with open("data.pkl",'wb') as wfp:
                pickle.dump(list_user_product, wfp)
        
        else:
            
            if len(list_user_product) == 0 :
                message = "Please give me your id "
            elif len(list_user_product) == 1 :
                message = "please give me the id of the product that you are looking for"
            elif len(list_user_product) == 2 :
                message = "How many products you need"
            else:
                        
                message, indexe = calculate_negociation(cmp_neg)
                with open("data.pkl",'rb') as rfp:
                    list_user_product= pickle.load(rfp)          
                products.at[indexe,"price_conversation"]=list_user_product[len(list_user_product)-1]
                products.to_csv("products.csv", index = False)
                print(products)

            
    return message 