from django.contrib import admin
from .models import City, Place, Route, Trip, Users, Plan, Tag


# Register your models here.
admin.site.register(City)
admin.site.register(Place)
admin.site.register(Route)
admin.site.register(Trip)
admin.site.register(Users)
admin.site.register(Plan)
admin.site.register(Tag)