from flask import Flask, render_template, request
import pickle


model = pickle.load(open('diabetic.pkl', 'rb'))

app=Flask(__name__)





@app.route('/')
def render():
    return render_template('input.html')

@app.route('/input', methods=['POST','GET'])
def input():
    #if request.method=="POST":

    

    msg=""
    detail=request.form
    age=detail['Age']
    gender=detail['Gender']
    polyuria=detail['Polyuria']
    polydipsia=detail['Polydipsia']
    suddenweightloss=detail['Sudden weight loss']
    weakness=detail['weakness']
    polyphagia=detail['Polyphagia']
    genitalthrush=detail['Genital thrush']
    visualblurring=detail['visual blurring']
    itching=detail['Itching']
    irritability=detail['Irritability']
    delayedhealing=detail['delayed healing']
    partialparesis=int(detail['partial paresis'])
    musclestiffness=int(detail['muscle stiffness'])
    alopecia = int(detail['Alopecia'])
    obesity = int(detail['Obesity'])
    outcome=[age,gender,polyuria,polydipsia,suddenweightloss,weakness,polyphagia,genitalthrush,visualblurring,itching,irritability,delayedhealing,partialparesis,musclestiffness,alopecia,obesity]
    print(outcome)
    outcome=standard.fit_transform([outcome])
    arr=[]
    for i in outcome[0]:
        arr.append(i)
   
    prediction=model.predict([arr])
    if prediction==0:
        msg="NO , you dont have the symptoms of  diabetes"
    else:
        msg="YES, you have the symptoms of diabetes  hence kindly check with your doctor "

    

    
    
    html = render_template(
    'output.html',
    msg=msg,
    
    
)
    return (html)