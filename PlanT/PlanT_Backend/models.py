# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models
import datetime


class Tag(models.Model):
    tag_id = models.PositiveSmallIntegerField(primary_key=True)
    tag_name = models.CharField(max_length=20)
    tag_time = models.PositiveSmallIntegerField()
    tag_co2 = models.PositiveSmallIntegerField()


class City(models.Model):
    city_id = models.PositiveSmallIntegerField(primary_key=True)
    city_name = models.CharField(max_length=10)
    city_tag = models.ForeignKey(Tag, on_delete=models.CASCADE)
    

class Place(models.Model):
    place_id = models.PositiveIntegerField(primary_key=True)
    place_name = models.CharField(max_length=100)
    place_address = models.CharField(max_length=100)
    place_latitude = models.DecimalField(max_digits=8, decimal_places=6)
    place_longitude = models.DecimalField(max_digits=9, decimal_places=6)
    place_eco = models.BooleanField(default=False)
    place_city = models.ForeignKey(City, on_delete=models.CASCADE)
    place_tag = models.ForeignKey(Tag, on_delete=models.CASCADE)


class Users(models.Model):
    user_id = models.CharField(primary_key=True, max_length=15)      
    user_pw = models.CharField(max_length=20)
    user_name = models.CharField(unique=True, null=True, max_length=45)
    user_email = models.EmailField(unique=True, max_length=45)        
    user_point = models.PositiveIntegerField(default=0)


class Trip(models.Model):
    trip_id = models.PositiveIntegerField(primary_key=True)
    trip_start = models.DateField()
    trip_end = models.DateField()
    STATE_CHOICES = (
        (1, '예정'),
        (2, '진행중'),
        (3, '완료'),
    )
    trip_state = models.PositiveSmallIntegerField(choices=STATE_CHOICES)
    ECOLEVEL_CHOICES = (
        (0, 'Level 0'),
        (1, 'Level 1'),
        (2, 'Level 2'),
        (3, 'Level 3'),
        (4, 'Level 4'),
        (5, 'Level 5'),
    )
    trip_ecolevel = models.PositiveSmallIntegerField(choices=ECOLEVEL_CHOICES)
    trip_posstime = models.PositiveSmallIntegerField()
    trip_user = models.ForeignKey(Users, on_delete=models.CASCADE)


class Plan(models.Model):
    plan_id = models.PositiveIntegerField(primary_key=True)
    plan_date = models.DateField()
    plan_trip = models.ForeignKey(Trip, on_delete=models.CASCADE)
    plan_city = models.ForeignKey(City, on_delete=models.CASCADE)


class Route(models.Model):
    route_id = models.PositiveIntegerField(primary_key=True)
    TYPE_CHOICES = (
        (1, '친환경 경로'),
        (2, '최단 경로'),
        (3, '지정 경로'),
    )
    route_type = models.PositiveSmallIntegerField(choices=TYPE_CHOICES)
    TRANSPORT_CHOICES = (
        (1, '도보'),
        (2, '자전거'),
        (3, '대중교통'),
        (4, '자동차'),
    )
    route_transport = models.PositiveSmallIntegerField(choices=TRANSPORT_CHOICES)
    route_starttime = models.DateTimeField()
    route_endtime = models.DateTimeField()
    route_time = models.PositiveSmallIntegerField()
    route_co2 = models.PositiveSmallIntegerField()
    route_detail = models.JSONField(blank=True, null=True, default=None)
    route_start = models.ForeignKey(Place, on_delete=models.CASCADE, related_name='route_start_place')
    route_end = models.ForeignKey(Place, on_delete=models.CASCADE, related_name='route_end_place')
    route_plan = models.ForeignKey(Plan, on_delete=models.CASCADE)
