# Generated by Django 2.0.13 on 2020-07-23 11:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='AutoMLDataModel',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('Age', models.FloatField(default=0)),
                ('Workclass', models.CharField(max_length=200)),
                ('EducationNum', models.FloatField(default=0)),
                ('MaritalStatus', models.CharField(max_length=200)),
                ('Occupation', models.CharField(max_length=200)),
                ('Relationship', models.CharField(max_length=200)),
                ('Race', models.CharField(max_length=200)),
                ('Sex', models.CharField(max_length=200)),
                ('CapitalGain', models.FloatField(default=0)),
                ('CapitalLoss', models.FloatField(default=0)),
                ('Hoursperweek', models.FloatField(default=0)),
                ('Country', models.CharField(max_length=200)),
            ],
            options={
                'db_table': 'automldata',
            },
        ),
    ]
