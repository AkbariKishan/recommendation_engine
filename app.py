from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def house_price_pred():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html')
    else:
        # reading username from form
        user_name = " ".join([str(x) for x in request.form.values()])


        #loading user_final_rating table (from recommendation engine)
        user_final_rating = load(open('user_final_rating.pkl', 'rb'))

        # loading sentiment analysis model
        model_senti = load(open('model_senti.pkl', 'rb'))

        # loading Product Name Vs Reveiw_text table
        data_P_R = load(open('data_P_R.pkl', 'rb'))

        # loading TFIDF vectorizer
        tfidf = load(open('tfidf.pkl', 'rb'))

        d = pd.DataFrame(user_final_rating.loc[user_name]).sort_values(by = user_name, ascending=False).head(20)
        # d = user_final_rating.loc[user_name].sort_values(ascending=False)[0:20]
        product_list = d.index.to_list()
        # product_list = user_final_rating[user_final_rating.index==user_name].sort_values(user_name, axis=1, ascending=False).columns.to_list()[0:20]

        tfidf_input = data_P_R[data_P_R.name.isin(product_list)]
        feat = tfidf.transform(tfidf_input['reviews_text'])
        X = pd.DataFrame(feat.toarray(), columns = tfidf.get_feature_names())

        preds = pd.DataFrame(model_senti.predict_proba(X)[:,1], index=X.index)[0].apply(lambda x: 1 if x>0.68 else 0)
        tfidf_input['predictions'] = preds

        #final result
        fin_res = pd.DataFrame(tfidf_input.groupby('name')['predictions'].mean()).sort_values(by='predictions', ascending=False).head().index.to_list()

        return render_template('index.html', Recommended_Products=fin_res)


if __name__ == "__main__":
    app.run(debug=True)
