from django.db import models

# Create your models here.


# Create your models here.
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'AutoUsers'



class AutoMLDataModel(models.Model):
    id  =models.IntegerField(primary_key=True)
    Age = models.FloatField(default=0)
    Workclass = models.CharField(max_length=200)
    EducationNum = models.FloatField(default=0)
    MaritalStatus = models.CharField(max_length=200)
    Occupation = models.CharField(max_length=200)
    Relationship = models.CharField(max_length=200)
    Race = models.CharField(max_length=200)
    Sex = models.CharField(max_length=200)
    CapitalGain= models.FloatField(default=0)
    CapitalLoss = models.FloatField(default=0)
    Hoursperweek = models.FloatField(default=0)
    Country= models.CharField(max_length=200)

    #def __str__(self):
        #return self.id

    class Meta:
        db_table = 'automldata'


class MyPredectionsModels(models.Model):
    id = models.IntegerField(primary_key=True)
    YearsExperience = models.FloatField(default=0)
    Salary = models.FloatField(default=0)

    def __str__(self):
        return self.id
    class Meta:
        db_table = 'mypredections'

class ModelPredectionStoreModels(models.Model):
    id = models.IntegerField(primary_key=True)
    username = models.CharField(max_length=150)
    email = models.CharField(max_length=150)
    acheiveaccuracy = models.FloatField()
    testsize = models.FloatField()
    cdata = models.DateTimeField(auto_now_add=True)