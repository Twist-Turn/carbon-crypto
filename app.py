from flask import Flask,render_template,request,jsonify
import hashlib as hl
import pickle
import numpy as np
import matplotlib as plt 

app=Flask(__name__)
file=open('model.pkl','rb')

regr=pickle.load(file)
file.close()

@app.route("/",methods=["GET","POST"])
def landing_page():
    return render_template("landing_index.html")


@app.route("/sign_index1.html",methods=["GET","POST"])
def sign_log():
    if True:
        return render_template("sign_index1.html")
    
    return render_template("landing_index.html")

@app.route("/contact_index.html",methods=["GET","POST"])
def contact():
    if True:
        return render_template("contact_index.html")
    return render_template("landing_index.html")




file=open("model1.pkl",'rb')
k_regressor=pickle.load(file)
file.close()

@app.route("/Validation form/index.html",methods=["GET","POST"])
def validate():
   
   if request.method == 'POST':
      result = request.form
      
      surface_area=float(result['energy_consumption'])
      
      input_hash=result['HASH']
      input_size=[surface_area]
      class Block:
         def __init__(self,data,prev_hash):
            self.data = data
            self.prev_hash = prev_hash
            self.hash = self.calc_hash()

         def calc_hash(self):
            sha = hl.sha256()
            sha.update(self.data.encode("utf-8"))
            return sha.hexdigest()
      class Blockchain:
            
         def __init__(self):
            self.hashlist=["0"]
            self.chain = [self.create_genesis_block()]
            
         def create_genesis_block(self):
            return Block("NULL Block","0")
            
         def add_block(self,data):
            prev_block = self.chain[-1]
            new_block = Block(data,prev_block.hash)
            self.chain.append(new_block)
            


      blockchain = Blockchain()

      
      predict=k_regressor.predict([input_size])[0][0]
    
      hash1 = blockchain.add_block(str(predict))
      print(blockchain.chain[-1].hash)
      blockchain.hashlist.append(blockchain.chain[-1].hash)
      ind=blockchain.hashlist.index(input_hash)
      hash_data=blockchain.chain[ind]

      return render_template("index.html",predict=predict, hash_key=blockchain.chain[-1].hash)
   return render_template("valid_index.html")
file=open('model.pkl','rb')

regr=pickle.load(file)
file.close()
@app.route("/Dashboard_page2/dist/index.html")
def dash():
   lis=[]
   if request.method=="POST":
        myDict=request.form
        engine=float(myDict['Engine'])
        cylinder=float(myDict['Cylinder'])
        Fuel_consumption=float(myDict['Fuel_consumption'])

        input_size = [engine]
        X=[engine]+[cylinder]
        Y=[Fuel_consumption]
        test_y_ = regr.predict([input_size])[0][0]*10.67
        #print(test_y_)
        return render_template('result1.html',EMI=round(test_y_))
   return render_template('dash_index.html')

   
    
if __name__=='__main__':
    app.run(debug=True)

