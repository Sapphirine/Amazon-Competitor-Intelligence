import streamlit as st
import pandas as pd
import numpy as np
# scraper 
import os
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
import sqlite3
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
#model
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForMaskedLM
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("./6895/model/")
import plotly.express as px

#scraper function
def scrape_review(asin):
    review_date = []
    review_text = []
    overall_rate = []
    helpful_vote = []
    # open website
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    product_asin = asin
    web = 'https://www.amazon.com/dp/' + product_asin
    driver.get(web)
    driver.implicitly_wait(5)
    # find all review button
    all_review_button= driver.find_element(By.XPATH,'//a[@data-hook="see-all-reviews-link-foot"]')
    all_review_button.click()
    # find all review cards
    items = WebDriverWait(driver,10).until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="a-section celwidget"]')))

    for item in items:

        # find review_text
        body = item.find_element(By.XPATH,'.//span[@data-hook="review-body"]/span')
        review_text.append(body.text)
        #find review_time and location
        date = item.find_element(By.XPATH,'.//span[@data-hook="review-date"]')
        review_date.append(date.text)
        #find overall_rate out of five stars
        rate = item.find_element(By.XPATH,'.//i[@data-hook="review-star-rating"]').get_attribute("class")
        overall_rate.append(int(rate[26:27]))
        #find # of people find review helpful
        helpful = item.find_element(By.XPATH,'.//span[@data-hook="helpful-vote-statement"]')
        helpful_vote.append(helpful.text.split()[0])
    
    driver.quit()
    data = {
            'text':review_text,
           'rate':overall_rate,
            'vote':helpful_vote,
            'time':review_date,
            }
    review = pd.DataFrame(data)
    return review


#helper functions

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis = 0)

def generate_logits(prompt,tokenizer,model):
    #print(prompt)
    inputs = tokenizer(prompt,return_tensors="pt",padding=True)
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
    mask_index = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    predicted_logit = logits[0][mask_index][0]
    #print(predicted_logit)
    return predicted_logit

def generate_prompt(usage, text,mask="<mask>",label = "good",target = "quality"):
    prompts = {
            "aspect": {
                "good" : f"{text}. Overall, customer like this product because {mask}",
                "bad" : f"{text}. Overall, customer hate this product because {mask}.",
            },
                    
            "score":{
                label: f"{text}.Overall, this product's {target} is {mask}.",
            }
    }
    return prompts[usage][label]

def verbalize(verbalizer):
    return tokenizer.encode(f" {verbalizer}",add_special_tokens=False)

def sort_logit(verbalizers,logits):
    result = {}
    for verbalizer in verbalizers:
        verb_id = verbalize(verbalizer)
        if len(verb_id) > 1:
            continue
        result[verbalizer] = logits[verb_id[0]]
    return [(k,v) for k,v in sorted(result.items(),key=lambda item:item[1])]

def sort_tuple(tup): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of 
    # sublist lambda has been used 
    tup.sort(key = lambda x: x[1]) 
    return tup 

def conclude(results,aspects,overall = "good"):
    best_aspect = np.argmax([result[-1][1] for result in results])
    Top3 = [result[-1] for result in results]
    #print(Top3)
    Top3 = [i[0] for i in sort_tuple(Top3)[-5:]]
    if overall == "good":
        return f"Overall, your product's best aspect is {aspects[best_aspect]},people love these features: {Top3}"
    else:
        return f"Overall, your product's not-so-good aspect is {aspects[best_aspect]},people didn't like these features: {Top3}"



def generate_conclusion(texts,label = "good"): # this needs to be a pd column of texts!
    if texts.empty:
        return f"woohoo,no {label} reviews!"
    sum_probs = []
    for i in texts:
        prompt = generate_prompt("aspect",i,label = label)
        logits = generate_logits(prompt,tokenizer,model)
        probabilities = F.softmax(logits,dim=-1)[0]
        sum_probs.append(probabilities)
    probability = torch.sum(torch.stack(sum_probs),dim=0)
    results = [sort_logit(verbalizer,probability) for verbalizer in verbalizers]
    #print(results)
    final_text = conclude(results,aspects,overall = label)
    return final_text

def generate_rating(logits):
    result = []
    for index in rating_indices:
        result.append(logits[index])
    return np.sum(softmax(result) * rating_scores)
#global stuff

quality = ['product','quality','expectation','material','advertised']
general = ['overall','great','good','perfect','bad','complaints','awesome']
price = ['price','cheap','expensive','value','afford','cost']
comments = ['daughter','wife','husband','friend','mom','recommend','thank']
experience = ["feel","smell","fit","comfort","clean","taste","scent","cute","pretty"]
ratings = ['great','good','average','bad','terrible']
verbalizers = [quality,general,price,comments,experience]
aspects =["quality","general","price","comments","experience"]
rating_indices = [verbalize(i)[0] for i in ratings]
rating_scores = np.array([5,4,3,2,1])


# main function of demo
def main():
    st.title('Amazon Product Competence Analysis')
    col1, col2= st.columns(2)

    with col1:
            # asin input
        with st.form("my_form1"):
            asin = st.text_input("Input ASIN:", value="B07Y26CQ7V", max_chars=10, key=None, type="default",help = "Amazon unique product ID")
            option = st.selectbox(
                'Which category your product belongs to?',
                ('Electronics', 'Beauty and Personal Care'))
            submit = st.form_submit_button("Get me the Analysis")
            
        # review output
        load_dotenv()
        
        if submit:
            #show top reviews
            review = scrape_review(asin)
            st.write('Top reviews')
            st.dataframe(review,width = None, height = None)

            #load data
            good_mask = (review["rate"] >= 3)
            bad_mask = (review["rate"] < 3)
            sum_probs = []
            #get conclusion first
            
            #print(f"bad text{review[bad_mask].text}")
            good_conclusion = generate_conclusion(review[good_mask].text)
            bad_conclusion = generate_conclusion(review[bad_mask].text,label="bad")
            st.write("*Advantages:*")
            st.write(good_conclusion)
            st.write("*Disadvantages:*")
            st.write(bad_conclusion)

            #sample: "Overall, your product's best aspect is quality,people love these features: ['material', 'product', 'quality']"
            #maybe save good_conclusion and bad_conclusions to txts, your call
            
            scores = []
            for text in review.text:
                score = []
                for i in verbalizers:
                    prompt = generate_prompt("score",text,target = i)
                    logits = generate_logits(prompt,tokenizer,model)
                    score.append(generate_rating(logits[0]))
                scores.append(score)
            scores = np.array(scores).sum(axis=0)/10 
            
            
            graph = pd.DataFrame(dict(
                r=scores,
                theta=aspects))
            fig = px.line_polar(graph, r='r', theta='theta', line_close=True,range_r = (0,5))
            fig.update_traces(fill='toself')
            st.write("*Download the graph for compatison!*")
            st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
            
        

    with col2:
        
        # asin input
        with st.form("my_form2"):
            asin = st.text_input("Input ASIN:", value="B07Y26CQ7V", max_chars=10, key=None, type="default",help = "Amazon unique product ID")
            option = st.selectbox(
                'Which category your product belongs to?',
                ('Electronics', 'Beauty and Personal Care'))
            submit = st.form_submit_button("Get me the Analysis")
            
        # review output
        load_dotenv()
        
        if submit:
            #show top reviews
            review = scrape_review(asin)
            st.write('Top reviews')
            st.dataframe(review,width = None, height = None)

            #load data
            good_mask = (review["rate"] >= 3)
            bad_mask = (review["rate"] < 3)
            sum_probs = []
            #get conclusion first
            
            #print(f"bad text{review[bad_mask].text}")
            good_conclusion = generate_conclusion(review[good_mask].text)
            bad_conclusion = generate_conclusion(review[bad_mask].text,label="bad")
            st.write("*Advantages:*")
            st.write(good_conclusion)
            st.write("*Disadvantages:*")
            st.write(bad_conclusion)

            #sample: "Overall, your product's best aspect is quality,people love these features: ['material', 'product', 'quality']"
            #maybe save good_conclusion and bad_conclusions to txts, your call
            
            scores = []
            for text in review.text:
                score = []
                for i in verbalizers:
                    prompt = generate_prompt("score",text,target = i)
                    logits = generate_logits(prompt,tokenizer,model)
                    score.append(generate_rating(logits[0]))
                scores.append(score)
            scores = np.array(scores).sum(axis=0)/10 
            
            
            graph = pd.DataFrame(dict(
                r=scores,
                theta=aspects))
            fig = px.line_polar(graph, r='r', theta='theta', line_close=True,range_r = (0,5))
            fig.update_traces(fill='toself')
            st.write("*Download the graph for compatison!*")
            st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
            
            

if __name__ == '__main__':
    main()






