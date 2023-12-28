import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import transformers
from transformers import DistilBertTokenizer
from transformers import TFDistilBertModel,TFBertModel
#============================================================
#讀取模型
#model = load_model('best_DistillBert_weights.h5')
#path2= 'best_Bert_weights.h5'
path = 'best_DistillBert_weights.h5'
model = load_model(path, custom_objects={'TFDistilBertModel': TFDistilBertModel})
print(model.summary())
#資料前處理
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased",do_lower_case=True) 

def bert_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []
  

    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
          data[i],
          add_special_tokens=True, # add  [CLS] or [SEP]
          max_length=maximum_length,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
        )
        
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)

#預測function
def pred(input):
    # Tokenize the input sentence into a word list, then encode each word in this list
    result_input_ids,result_attention_masks = bert_encode([input],64)

    # To tesnor format
    input_ids = tf.constant(result_input_ids)
    attention_mask = tf.constant(result_attention_masks)

    # Feed the encoded lsit into model and predict 
    label_map = {0: "Negative", 1: "Positive"}
    prob = model([input_ids, attention_mask])
    predicted_labels = (prob.numpy() > 0.5).astype(int)

    for i, label in enumerate(predicted_labels):
      print("Predicted label:", label_map[label[0]], "\n")
    

    return label_map[label[0]]


#test1 = "Chefs are nutters. They’re all self-obsessed, delicate, dainty, insecure little souls, and absolute psychopaths. Every last one of them. My gran could do better! And she’s dead!"
#test2 = "This dish, my dear friend, is nothing short of a culinary catastrophe. I've never encountered anything quite so disastrous in all my years in the kitchen. The flavors are an absolute train wreck, and the texture is reminiscent of a soggy, overcooked sponge. If this were a competition, it would undoubtedly be a contestant for the 'Worst Dish of the Century' award. An absolute travesty to the world of cuisine!"
#test3 = "The culinary delights on offer here are simply extraordinary, with each bite offering an explosion of exquisite flavors that dance harmoniously on the palate, making it an unforgettable and truly gastronomic experience."

#print(pred(test3))


