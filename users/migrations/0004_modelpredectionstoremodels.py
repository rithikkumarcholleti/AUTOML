# Generated by Django 2.0.13 on 2020-07-27 06:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_mypredectionsmodels'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelPredectionStoreModels',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('username', models.CharField(max_length=150)),
                ('email', models.CharField(max_length=150)),
                ('acheiveaccuracy', models.FloatField()),
                ('testsize', models.FloatField()),
                ('cdata', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
