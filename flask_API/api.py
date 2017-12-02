"""Cloud Foundry test"""
from flask import Flask, render_template, url_for
import os
import psycopg2
import pickle
import numpy as np
from flask_cache import Cache
from random import sample

try:
    conn = psycopg2.connect("dbname='compose' user='admin' host='sl-us-south-1-portal.14.dblayer.com' password='YKNEUFLDTRFNUTRU' port=29203")
    cur = conn.cursor()


except:
    print ("I am unable to connect to the database")


model_packed = pickle.load(open("model_packed.pkl", "rb"))
embeddings_eval, nce_weights_eval, nce_biases_eval, dictionary, reversed_dictionary = model_packed

DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)
port = int(os.getenv("PORT"))

cache = Cache(app,config={'CACHE_TYPE': 'simple'})

# @app.route('/')
def hello_world():
    return 'Hello World! I am Nuck Chorris. The best team.'

@app.route("/user/<string:id>/<int:count>")
@app.route("/user/<string:id>")
@cache.memoize(timeout=500)
def user(id, count=1):
    cur.execute("SELECT product_id, title_full_c, variant_id FROM user_views WHERE user_id='%s' ORDER BY id;"%id)
    products = cur.fetchall()[:count]
    print(products)
    viewed = [(image_url(str(prod[0]), str(prod[2])), prod[1]) for prod in products]
    products = [str(prod[0]) for prod in products]
    top = items_in_order(products)[:48]
    excepted = [(image_url(id, "1"), prod_name(id)) for id in top]
    print(viewed)
    print(excepted)

    return render_template('user.html', viewed=viewed, excepted=excepted, next=count+1, previous=count-1, user=id)


@app.route("/recommendation/<string:ids>")
@cache.memoize(timeout=500)
def interactive(ids):
        products=ids.split(",")[1:]
        print(products)
        viewed = [(image_url(str(id), "1"), prod_name(id), id) for id in products]
        products = [str(prod) for prod in products]
        top = items_in_order(products)[:48]
        excepted = [(image_url(id, "1"), prod_name(id), id) for id in top]
        print(viewed)
        print(excepted)
        return render_template('random.html', viewed=viewed, excepted=excepted, ids=ids)


@app.route("/")
def random():
    print("random")
    # cur.execute("SELECT product_id, title_full_c, variant_id FROM user_views ORDER BY RANDOM() LIMIT 15;")
    # products = cur.fetchall()
    products = sample(random_values, 48)
    print("random finished")
    viewed=[]
    print("excepted")
    excepted = [(image_url(prod[0], prod[2]), prod[3], str(prod[0])) for prod in products]
    print(excepted)
    print("excepted finished")
    return render_template('random.html', viewed=viewed, excepted=excepted, ids="")

@app.route("/image/<int:id>")
def image(id):
    return render_template('image.html')

@cache.memoize(timeout=500)
def image_url(id, variant):
    print("image url %s"%id)
    cur.execute("SELECT img FROM PRODUCT2 WHERE id='%s' LIMIT 1;"%(variant))
    values = cur.fetchall()
    if len(values)>0:
        return values[0][0]
    else:
        id_int = int(id)
        id_low = id_int*1000
        id_high = id_low+999
        cur.execute("SELECT img FROM PRODUCT2 WHERE id='%s' OR (id>='%s' AND id<='%s') LIMIT 1;"%(id,id_low,id_high))
        values = cur.fetchall()
        if len(values)>0:
            return values[0][0]
        else:
            return ""

@cache.memoize(timeout=500)
def prod_name(id):
    print("prod name %s"%id)
    cur.execute("SELECT title_full_c FROM user_views WHERE product_id='%s' LIMIT 1;"%id)
    values = cur.fetchall()
    if len(values)>0:
        return values[0][0]
    else:
        return ""

@app.route("/version")
def version():
    return str(sys.version_info)

def items_in_order(item_list):
    ind_list = [dictionary[it] for it in item_list]
    mean_emb = np.mean(embeddings_eval[ind_list,:], axis=0)
    preferences = softmax(np.sum(nce_weights_eval * mean_emb, axis=1) + nce_biases_eval)
    order_presented = len(preferences) - np.argsort(np.argsort(preferences))
    items_in_order = [reversed_dictionary[i] for i in np.argsort(order_presented)]
    return items_in_order

def softmax(x):
    e_x = np.exp(x - np.max(x))
    softmax_x = e_x / e_x.sum()
    return softmax_x

if __name__ == '__main__':
    cur.execute("SELECT product_id, title_full_c, variant_id, title_full_c  FROM user_views ORDER BY RANDOM() LIMIT 2000;")
    random_values = [v for v in cur.fetchall()]
    for random in random_values:
        image_url(random[0], random[2])
        prod_name(random[0])
    app.run(host='0.0.0.0', port=port)