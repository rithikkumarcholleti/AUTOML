from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.forms import UserRegistrationModel
from users.models import ModelPredectionStoreModels
# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def AdminHome(request):
    return render(request,'admins/AdminHome.html',{})

def ViewUsersList(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/RegisteredUsers.html',{'data':data})

def UserPerformedOperations(request):
    data = ModelPredectionStoreModels.objects.all()
    return render(request, 'admins/UsersOperations.html', {'data': data})
