from flask import Flask, render_template, flash, request, session,send_file
from flask import render_template, redirect, url_for, request
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
import datetime
import mysql.connector
import sys

import pickle
import cv2

import numpy as np


app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def homepage():

    return render_template('index.html')

@app.route("/AdminLogin")
def AdminLogin():

    return render_template('AdminLogin.html')


@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')

@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')

@app.route("/NewQuery1")
def NewQuery1():
    return render_template('NewQueryReg.html')

@app.route("/UploadDataset")
def UploadDataset():
    return render_template('ViewExcel.html')



@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb ")
    data = cur.fetchall()
    return render_template('AdminHome.html',data=data)






@app.route("/UserHome")
def UserHome():
    user = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where username='" + user + "'")
    data = cur.fetchall()
    return render_template('UserHome.html',data=data)


@app.route("/UQueryandAns")
def UQueryandAns():

    uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult='waiting'")
    data = cur.fetchall()

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult !='waiting'")
    data1 = cur.fetchall()


    return render_template('UserQueryAnswerinfo.html', wait=data, answ=data1 )


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
       if request.form['uname'] == 'admin' or request.form['password'] == 'admin':

           conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
           # cursor = conn.cursor()
           cur = conn.cursor()
           cur.execute("SELECT * FROM regtb ")
           data = cur.fetchall()
           return render_template('AdminHome.html' , data=data)

       else:
        return render_template('index.html', error=error)


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():

    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            alert = 'Username or Password is wrong'
            render_template('goback.html', data=alert)



        else:
            print(data[0])
            session['uid'] = data[0]
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and Password='" + password + "'")
            data = cur.fetchall()

            return render_template('UserHome.html', data=data )




@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':

        name1 = request.form['name']
        gender1 = request.form['gender']
        Age = request.form['age']
        email = request.form['email']
        pnumber = request.form['phone']
        address = request.form['address']

        uname = request.form['uname']
        password = request.form['psw']


        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO regtb VALUES ('" + name1 + "','" + gender1 + "','" + Age + "','" + email + "','" + pnumber + "','" + address + "','" + uname + "','" + password + "')")
        conn.commit()
        conn.close()
        # return 'file register successfully'


    return render_template('UserLogin.html')



@app.route("/newquery", methods=['GET', 'POST'])
def newquery():
    if request.method == 'POST':
        uname = session['uname']
        nitrogen = request.form['nitrogen']
        phosphorus = request.form['phosphorus']
        potassium = request.form['potassium']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']
        location = request.form['select']




        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Querytb VALUES ('','" + uname + "','" + nitrogen + "','" + phosphorus + "','" + potassium + "','"+temperature+"','"+humidity +"','"+ ph
            +"','"+ rainfall +"','waiting','','','"+location+"')")
        conn.commit()
        conn.close()
        # return 'file register successfully'
        uname = session['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult='waiting'")
        data = cur.fetchall()

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult !='waiting'")
        data1 = cur.fetchall()

        return render_template('UserQueryAnswerinfo.html', wait=data, answ=data1)


@app.route("/excelpost", methods=['GET', 'POST'])
def uploadassign():
    if request.method == 'POST':


        file = request.files['fileupload']
        file_extension = file.filename.split('.')[1]
        print(file_extension)
        #file.save("static/upload/" + secure_filename(file.filename))

        import pandas as pd
        import matplotlib.pyplot as plt
        df = ''
        if file_extension == 'xlsx':
            df = pd.read_excel(file.read(), engine='openpyxl')
        elif file_extension == 'xls':
            df = pd.read_excel(file.read())
        elif file_extension == 'csv':
            df = pd.read_csv(file)



        print(df)




        import seaborn as sns
        sns.countplot(df['label'], label="Count")
        plt.savefig('static/images/out.jpg')
        iimg = 'static/images/out.jpg'

        #plt.show()

        #df = pd.read_csv("./Heart/Heartnew.csv")

        #def clean_dataset(df):
            #assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
            #df.dropna(inplace=True)
            #indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
            #return df[indices_to_keep].astype(np.float64)

        #df = clean_dataset(df)

        #print("Preprocessing Completed")
        print(df)




        # import pandas as pd
        import matplotlib.pyplot as plt

        # read-in data
        # data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs

        import seaborn as sns
        sns.countplot(df['label'], label="Count")
        plt.show()

        df.label = df.label.map({'rice': 0,
                                 'maize': 1,
                                 'chickpea': 2,
                                 'kidneybeans': 3,
                                 'pigeonpeas': 4,
                                 'mothbeans': 5,
                                 'mungbean': 6,
                                 'blackgram': 7,
                                 'lentil': 8,
                                 'pomegranate': 9,
                                 'banana': 10,
                                 'mango': 11,
                                 'grapes': 12,
                                 'watermelon': 13,
                                 'muskmelon': 14,
                                 'apple': 15,
                                 'orange': 16,
                                 'papaya': 17,
                                 'coconut': 18,
                                 'cotton': 19,
                                 'jute': 20,
                                 'coffee': 21})

        # Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
        df_copy = df.copy(deep=True)
        df_copy[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] = df_copy[
            ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].replace(0, np.NaN)

        # Model Building
        from sklearn.model_selection import train_test_split
        df.drop(df.columns[np.isnan(df).any()], axis=1)
        X = df.drop(columns='label')
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import classification_report
        classifier = MLPClassifier(random_state=0)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        clreport = classification_report(y_test, y_pred)

        print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

        Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
        Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))



        # Creating a pickle file for the classifier
        filename = 'crop-prediction-rfc-model.pkl'
        pickle.dump(classifier, open(filename, 'wb'))



        print("Training process is complete Model File Saved!")

        df= df.head(200)

        #read_csv(..., skiprows=1000000, nrows=999999)



        return render_template('ViewExcel.html', data=df.to_html(), dataimg=iimg ,tacc=Tacc,testacc=Testacc,report=clreport)


@app.route("/AdminQinfo")
def AdminQinfo():

    #uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult='waiting'")
    data = cur.fetchall()


    return render_template('AdminQueryInfo.html', data=data )


@app.route("/answer")
def answer():

    Answer = ''
    Prescription=''
    id =  request.args.get('lid')


    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM Querytb where  id='" + id + "'")
    data = cursor.fetchone()

    if data:
        UserName = data[1]
        nitrogen = data[2]
        phosphorus = data[3]
        potassium = data[4]
        temperature = data[5]
        humidity = data[6]
        ph = data[7]
        rainfall = data[8]



    else:
        return 'Incorrect username / password !'

    nit = float(nitrogen)
    pho = float(phosphorus)
    po = float(potassium)
    te = float(temperature)
    hu = float(humidity)
    phh = float(ph)
    ra = float(rainfall)
    #age = int(age)

    filename = 'crop-prediction-rfc-model.pkl'
    classifier = pickle.load(open(filename, 'rb'))

    data = np.array([[nit, pho, po, te, hu, phh, ra ]])
    my_prediction = classifier.predict(data)
    print(my_prediction)


    crop = ''
    fertilizer = ''

    if my_prediction == 0:
        Answer = 'Predict'
        crop = 'rice'

        fertilizer ='4 kg of gypsum and 1 kg of DAP/cent can be applied at 10 days after sowing'

    elif my_prediction == 1:
        Answer = 'Predict'
        crop = 'maize'
        fertilizer = 'The standard fertilizer recommendation for maize consists of 150 kg ha−1 NPK 14–23–14 and 50 kg ha−1 urea'
    elif my_prediction == 2:
        Answer = 'Predict'
        crop = 'chickpea'

        fertilizer = 'The generally recommended doses for chickpea include 20–30 kg nitrogen (N) and 40–60 kg phosphorus (P) ha-1. If soils are low in potassium (K), an application of 17 to 25 kg K ha-1 is recommended'

    elif my_prediction == 3:
        Answer = 'Predict'
        crop = 'kidneybeans'
        fertilizer = 'It needs good amount of Nitrogen about 100 to 125 kg/ha'

    elif my_prediction == 4:
        Answer = 'Predict'
        crop = 'pigeonpeas'
        fertilizer = 'Apply 25 - 30 kg N, 40 - 50 k g P 2 O 5 , 30 kg K 2 O per ha area as Basal dose at the time of sowing.'

    elif my_prediction == 5:
        Answer = 'Predict'
        crop = 'mothbeans'
        fertilizer = 'The applications of 10 kg N+40 kg P2O5 per hectare have proved the effective starter dose'
    elif my_prediction == 6:
        Answer = 'Predict'
        crop = 'mungbean'
        fertilizer = 'Phosphorus and potassium fertilizers should be applied at 50-50 kg ha-1'
    elif my_prediction == 7:
        Answer = 'Predict'
        crop = 'blackgram'
        fertilizer = 'The recommended fertilizer dose for black gram is 20:40:40 kg NPK/ha.'
    elif my_prediction == 8:
        Answer = 'Predict'
        crop = 'lentil'
        fertilizer = 'The recommended dose of fertilizers is 20kg N, 40kg P, 20 kg K and 20kg S/ha.'
    elif my_prediction == 9:
        Answer = 'Predict'
        crop = 'pomegranate'
        fertilizer = 'The recommended fertiliser dose is 600–700 gm of N, 200–250 gm of P2O5 and 200–250 gm of K2O per tree per year'

    elif my_prediction == 10:
        Answer = 'Predict'
        crop = 'banana'
        fertilizer = 'Feed regularly using either 8-10-8 (NPK) chemical fertilizer or organic composted manure'

    elif my_prediction == 11:
        Answer = 'Predict'
        crop = 'mango'
        fertilizer = '50 gm zinc sulphate, 50 gm copper sulphate and 20 gm borax per tree/annum are recommended'

    elif my_prediction == 12:
        Answer = 'Predict'
        crop = 'grapes'
        fertilizer = 'Use 3 pounds (1.5 kg.) of potassium sulfate per vine for mild deficiencies or up to 6 pounds (3 kg.)'

    elif my_prediction == 13:
        Answer = 'Predict'
        crop = 'watermelon'
        fertilizer = 'Apply a fertilizer high in phosphorous, such as 10-10-10, at a rate of 4 pounds per 1,000 square feet (60 to 90 feet of row)'

    elif my_prediction == 14:
        Answer = 'Predict'
        crop = 'muskmelon'
        fertilizer = 'Apply FYM 20 t/ha, NPK 40:60:30 kg/ha as basal and N @ 40 kg/ha 30 days after sowing.'

    elif my_prediction == 15:
        Answer = 'Predict'
        crop = 'apple'
        fertilizer = 'Apple trees require nitrogen, phosphorus and potassium,Common granular 20-10-10 fertilizer is suitable for apples'

    elif my_prediction == 16:
        Answer = 'Predict'
        crop = 'orange'
        fertilizer = 'Orange farmers often provide 5,5 – 7,7 lbs (2,5-3,5 kg) P2O5 in every adult tree for 4-5 consecutive years'

    elif my_prediction == 17:
        Answer = 'Predict'
        crop = 'papaya'
        fertilizer = 'Generally 90 g of Urea, 250 g of Super phosphate and 140 g of Muriate of Potash per plant are recommended for each application'

    elif my_prediction == 18:
        Answer = 'Predict'
        crop = 'coconut'
        fertilizer = 'Organic Manure @50kg/palm or 30 kg green manure, 500 g N, 320 g P2O5 and 1200 g K2O/palm/year in two split doses during September and May'

    elif my_prediction == 19:
        Answer = 'Predict'
        crop = 'cotton'
        fertilizer = 'N-P-K 20-10-10 per hectare during sowing (through the sowing machine)'

    elif my_prediction == 20:
        Answer = 'Predict'
        crop = 'jute'
        fertilizer = 'Apply 10 kg of N at 20 - 25 days after first weeding and then again on 35 - 40 days after second weeding as top dressing'

    elif my_prediction == 21:
        Answer = 'Predict'
        crop = 'coffee'
        fertilizer = 'Coffee trees need a lot of potash, nitrogen, and a little phosphoric acid. Spread the fertilizer in a ring around each Coffee plant'


    else:
        Answer = 'Predict'
        crop='Crop info not Found!'







    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cursor = conn.cursor()
    cursor.execute(
        "update Querytb set DResult='"+Answer+"', CropInfo='" + crop +"',Fertilizer='"+fertilizer+"' where id='" + str(id) + "' ")
    conn.commit()
    conn.close()

    conn3 = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cur3 = conn3.cursor()
    cur3.execute("SELECT * FROM regtb where 	UserName='" + str(UserName) + "'")
    data3 = cur3.fetchone()
    if data3:
        phnumber = data3[4]
        print(phnumber)
        sendmsg(phnumber, "Predict Crop Name : "+ crop +" For More info Visit in Site")

    # return 'file register successfully'
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult !='waiting '")
    data = cur.fetchall()
    return render_template('AdminAnswer.html', data=data)



@app.route("/AdminAinfo")
def AdminAinfo():



    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult !='waiting'")
    data = cur.fetchall()


    return render_template('AdminAnswer.html', data=data )


@app.route("/Soli")
def Soli():

    return render_template('Soli.html' )

def sendmsg(targetno,message):
    import requests
    requests.post("http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=" + targetno + "&from=FSSMSS&message=Dear user  your msg is " + message + " Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640")


@app.route("/testimage", methods=['GET', 'POST'])
def testimage():
    if request.method == 'POST':


        file = request.files['fileupload']
        file.save('static/Out/Test.jpg')

        img = cv2.imread('static/Out/Test.jpg')
        if img is None:
            print('no data')

        img1 = cv2.imread('static/Out/Test.jpg')
        print(img.shape)
        img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
        original = img.copy()
        neworiginal = img.copy()
        cv2.imshow('original', img1)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img1S = cv2.resize(img1, (960, 540))

        cv2.imshow('Original image', img1S)
        grayS = cv2.resize(gray, (960, 540))
        cv2.imshow('Gray image', grayS)

        gry = 'static/Out/gry.jpg'

        cv2.imwrite(gry, grayS)
        from PIL import  ImageOps,Image

        im = Image.open(file)

        im_invert = ImageOps.invert(im)
        inv = 'static/Out/inv.jpg'
        im_invert.save(inv, quality=95)

        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        cv2.imshow("Nosie Removal", dst)
        noi = 'static/Out/noi.jpg'

        cv2.imwrite(noi, dst)

        import warnings
        warnings.filterwarnings('ignore')

        import tensorflow as tf
        classifierLoad = tf.keras.models.load_model('soalmodel.h5')

        import numpy as np
        from keras.preprocessing import image

        test_image = image.load_img('static/Out/Test.jpg', target_size=(200, 200))
        img1 = cv2.imread('static/Out/Test.jpg')
        # test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)

        out = ''
        pre = ''
        if result[0][0] == 1:

            out = "AlluvialSoil"
            pre ="Wheat, Groundnut and cotton"

        elif result[0][1] == 1:

            out = "ClaySoil "
            pre = "Cabbage (Napa and savoy), Cauliflower, Kale, Bean, Pea, Potato and Daikon radish"

        elif result[0][2] == 1:

            out = "RedSoil"
            pre = "Marsh soils are not suitable for crop cultivation due to their high acidic nature"

        elif result[0][3] == 1:

            out = "YellowSoil "
            pre = "Tea, coffee and cashew "




        org = 'static/Out/Test.jpg'
        gry ='static/Out/gry.jpg'
        inv = 'static/Out/inv.jpg'
        noi = 'static/Out/noi.jpg'




        return render_template('Soli.html',result=out,org=org,gry=gry,inv=inv,noi=noi,pre=pre)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)