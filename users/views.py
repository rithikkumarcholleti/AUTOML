from django.shortcuts import render,HttpResponse,redirect
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel,AutoMLDataModel,MyPredectionsModels,ModelPredectionStoreModels
from .UserAutoMachineLearningProcess import StartProcessAutoML
import h2o
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import csv,io
from django_pandas.io import read_frame
import matplotlib.pyplot as plt
import numpy as np
# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UsersRegister.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UsersRegister.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def UserAutoMLTest(request):
    obj = StartProcessAutoML()
    html = ''
    data1=''
    try:

        pass
        lb = obj.startDataPreprocess()
        data_as_df = h2o.as_list(lb)
        html = data_as_df.to_html()
        #data1 = data.to_html()

    except Exception as e:
        pass
    data_list = AutoMLDataModel.objects.all()
    #print("Lb type is ",type(lb))
    return render(request,"users/AutoMachineLearning.html",{"html":html,"dataset":data_list})
    #return HttpResponse("Exit code 0")
    #return redirect('AutoResponse')

def AutoResponse(request):
    data_list = AutoMLDataModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 10)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)
    return render(request, 'users/AutoMachineLearning.html', {'users': users})

def DataUploadForm(request):
    return render(request,'users/useruploaddata.html',{})

def UploadDatatoServer(request):
    AutoMLDataModel
    # declaring template
    template = "users/useruploaddata.html"
    data = AutoMLDataModel.objects.all()
    # prompt is a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be name, email, address,    phone, profile',
        'profiles': data
    }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')

    # setup a stream which is when we loop through each line we are able to handle a data in a stream
    io_string = io.StringIO(data_set)
    next(io_string)
    for column in csv.reader(io_string, delimiter='\t', quotechar="|"):
        print("Data is = ",column[0])
        _, created = AutoMLDataModel.objects.update_or_create(
            Age=column[1],
            Workclass=column[2],
            EducationNum=column[3],
            MaritalStatus=column[4],
            Occupation=column[5],
            Relationship=column[6],
            Race=column[7],
            Sex=column[8],
            CapitalGain=column[9],
            CapitalLoss=column[10],
            Hoursperweek=column[11],
            Country=column[12]


        )
    context = {}

    return render(request, 'users/useruploaddata.html', context)

def UploadDatatoServerForPredections(request):
    csv_file = request.FILES['file']
    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')
    # setup a stream which is when we loop through each line we are able to handle a data in a stream
    io_string = io.StringIO(data_set)
    next(io_string)
    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        print("Data is = ", column[0])
        _, created = MyPredectionsModels.objects.update_or_create(
            YearsExperience=column[0],
            Salary=column[1]

        )
    context = {}

    return render(request, 'users/useruploaddata.html', context)

def MyPredectionsSlot1(request):
    data = MyPredectionsModels.objects.all()
    return render(request,'users/MyPredections.html',{'data':data})

def MyPredectionsSlot2(request):
    data = MyPredectionsModels.objects.all()
    return render(request, 'users/DataSlot1.html', {'data': data})


def MyPredectionsSlot3(request):
    if request.method=='POST':
        splitsize = int(request.POST.get('testsize'))
        testsize = splitsize/100
        data = MyPredectionsModels.objects.all()
        dataset = read_frame(data)
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, -1].values
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=0)
        #print('X_train', X_train)
        #print('X_test', X_test)
        #print('y_train', y_train)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)

        plt.scatter(X_test, y_test, color='red')
        plt.plot(X_train, model.predict(X_train), color='blue')
        plt.title('Salary vs Experience (Test set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()
        score =  model.score(X, y)
        loginid = request.session['loginid']
        email = request.session['email']
        ModelPredectionStoreModels.objects.create(username=loginid, email=email, acheiveaccuracy=score,testsize=testsize)
        y_pred = model.predict(X_test)
        y_pred = np.around(y_pred, 1)
        print('predected Result ', type(y_pred.tolist()))
        print('Original salary ', type(y_test))
        myDict = {'original':y_test.tolist(),'predections':y_pred.tolist()}
        print("My Dict ",myDict)
    return render(request,'users/DataSlot2.html',{'data':myDict})
    #return HttpResponse(html)

def AddDataForm(request):
    data = MyPredectionsModels.objects.all()
    return render(request, 'users/AddDataForm.html', {'data': data})


def AddDataToDataset(request):
    if request.method=='POST':
        exp = request.POST.get('Experience')
        salary = request.POST.get('salary')
        MyPredectionsModels.objects.create(YearsExperience=exp,Salary=salary)
        data = MyPredectionsModels.objects.all()
        return render(request, 'users/DataSlot1.html', {'data': data})