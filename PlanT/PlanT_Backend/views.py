from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from .test_serializer import MyTokenObtainPairSerializer, RegisterSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import generics
from django.contrib.auth.models import User
from rest_framework.permissions import AllowAny, IsAuthenticated
from .models import Users, City, Place, Plan, Trip, Route, Tag
from django.views.decorators.csrf import csrf_exempt
import json
#from django.core.serializers import serialize

# Create your views here.


@csrf_exempt
def sender(request):
    city = City.objects.all().values_list("city_name", flat=True)
#    plc1 = Place.objects.all().values_list("city_id", flat=True)
#    plc2 = Place.objects.all().values_list("city_id", flat=True)
#    Plan = City.objects.all().values_list("city_id", flat=True)
    
    return JsonResponse(list(city), safe=False)


@csrf_exempt
def reciever(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))  # JSON 데이터를 파이썬 객체로 변환
        option_value = data.get('ops')
        if option_value:
            option = str(option_value)
            cid = City.objects.get(city_name=option)
            option = Place.objects.filter(city=cid).values_list('place_id','place_name')
            dict = {}
            for i in option:
                dict[i[0]] = i[1]
            #serialized_option = serialize('json', option, fields=('place_name',))
            return JsonResponse(dict, safe=False)
        else:
            return JsonResponse({'error': 'Option value is required'}, status=400)
        
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)



class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer


@api_view(['GET'])
def getRoutes(request):
    routes = [
        '/api/token/',
        '/api/register/',
        '/api/token/refresh/'
    ]
    return Response(routes)


class recommender:
    def __init__(self, ):



def classifier(plan_dict):
    tag_list = Tag.objects.values_list('tag_id', flat=True)
    tag_dict = {}
    
    for tag in tag_list:
        tag_dict[tag] = 0
            
    plc = {}
    pln = {}
        
    for i in range(1,len(plan_dict)+1):
        plc_l = []
        pln_l = []
        for j in plan_dict[i]:
            if j in tag_list:
                pln_l.append(j)
                tag_dict[j] += 1
            else:
                plc_l.append(j)
        plc[i] = plc_l
        pln[i] = pln_l
        
    
       
    return [tag_dict, plc, pln]   
          
            
def assigntag(eco_lev, tag_dict):
    if eco_lev == 0:
        tag_per = 0
    elif eco_lev == 1:
        tag_per = 20
    elif eco_lev == 2:
        tag_per = 40
    elif eco_lev == 3:
        tag_per = 60
    elif eco_lev == 4:
        tag_per = 80
    else:
        tag_per = 100
        
    for i in tag_dict:
        tag_dict[i] /= eco_lev
        
    return tag_dict
             
            
def planner(plan_dict, tag_dict, pln, plc):
    for i in range(1,len(plan_dict)+1):
        if pln:
            for j in range(1,len(plan_dict[i])):
                if plan_dict[i][j] - 10 < 0 <= plan_dict[i][j - 1] - 10:
                    l = [i,plan_dict[i][j-1]]
                    # l 보내서 l로부터 접근 가능한 장소(id) 리스트로 받아오기
                    plc_can = Place.objects.filter(place_tag = plan_dict[i][j]) # 나중에 받으면 교체
                    if tag_dict[plan_dict[i][j]] == 0:
                        plc_can = plc_can.filter(place_eco=False)                    
                    l = plc_can.values_list('place_id', flat=True)
                    # l 보내서 가장 탄소배출량 적은 장소(id),이동수단,소요시간,탄소배출량 받기
                    plc_id = 11
                    plc_move = 3
                    plc_time = 30
                    plc_co2 = 30
                    if Place.objects.get(pk=plc_id).place_eco:
                        tag_dict[plan_dict[i][j]] -= 1
                    
                    

            




