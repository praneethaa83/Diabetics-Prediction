from flask import Flask, render_template, request
import pickle


model = pickle.load(open('dbt.pkl', 'rb'))

app=Flask(__name__)





@app.route('/')
def render():
    return render_template('input.html')

@app.route('/input', methods=['POST','GET'])
def input():
    #if request.method=="POST":

    

    msg=""
    detail=request.form
    age=detail['age']
    gender=detail['gender']
    polyuria=detail['polyuria']
    polydipsia=detail['polydipsia']
    suddenweightloss=detail['suddenweightloss']
    weakness=detail['weakness']
    polyphagia=detail['polyphagia']
    genitalthrush=detail['Genitalthrush']
    visualblurring=detail['visionblurring']
    itching=detail['itching']
    irritability=detail['irritability']
    delayedhealing=detail['delayedhealing']
    partialparesis=detail['pp']
    musclestiffness=detail['musclestiffness']
    alopecia = detail['alopecia']
    obesity = detail['obesity']
    outcome=[age,gender,polyuria,polydipsia,suddenweightloss,weakness,polyphagia,genitalthrush,visualblurring,itching,irritability,delayedhealing,partialparesis,musclestiffness,alopecia,obesity]
    print(outcome)
   
    prediction=model.predict([outcome])
    if prediction==0:
        msg="NO , you dont have the symptoms of  diabetes"
    else:
        msg="YES, you have the symptoms of diabetes  hence kindly check with your doctor "

    

    
    
    html = render_template(
    'output.html',
    msg=msg,
    
    
)
    return (html)


if __name__=='__main__':
    app.run(debug=True)