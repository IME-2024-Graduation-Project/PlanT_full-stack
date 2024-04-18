import pandas as pd
import math
import os
from ortools.linear_solver import pywraplp
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import requests
import json



# place id

# place id
pppp = {
    1:[i for i in range(123,137)] ,
    2:[i for i in range(34,48)] ,
    3:[1, 25, 36, 49, 151, 221, 121, 641, 222, 526, 556, 94, 111, 131]
}


def cluster(cluster_count,input):
    #변경
    df2 = pd.DataFrame()

    df2['place_id'] = input[3]
    df2['place_longitude'] = input[2]
    df2['place_latitude'] = input[1]


    num1 = input[3]
    d = cluster_count
    def data(input):
        

        sample_x = []
        sample_y = []

        num = input[3]


        
        for j in range(len(num)):
                sample_x.append(input[2][j]*10000)
                sample_y.append(input[2][j]*10000 )

        #1자
        sample_dist = []
        #리스트 속 리스트.
        sample_dist_2 = []

        for k in range(len(sample_x)):
            for l in range(len(sample_y)):
                if k == l :
                    sample_dist.append(9999999999)

                elif k != l:
                    sample_dist.append(abs(sample_x[k]-sample_x[l] + sample_y[k]-sample_y[l]))


        for k in range(len(sample_x)):
            data = []
            for l in range(len(sample_y)):
            
                if k == l :
                    data.append(0)

                elif k != l:
                    data.append(abs(sample_x[k]-sample_x[l] + sample_y[k]-sample_y[l]))

            sample_dist_2.append(data)

        return {'li_in_li': sample_dist_2 ,
                'one' : sample_dist
                }
    

    data1 = data(input)['li_in_li']

    data3 = df2[['place_id','place_longitude', 'place_latitude']]

    '''
    K-means
    '''
    def k_means_clust(k,da):
            
        

            # 두 가지 feature를 대상
            data = da[['place_id','place_longitude', 'place_latitude']]#.iloc[[0,24,35,48,150,220,120,640]]

            #data = dat[0:15]
            

            # 정규화 진행
            scaler = MinMaxScaler()
            data_scale = scaler.fit_transform(data)

            # 그룹 수, random_state 설정
            model = KMeans(n_clusters = k, random_state = 10)

            # 정규화된 데이터에 학습
            model.fit(data_scale)

            # 클러스터링 결과 각 데이터가 몇 번째 그룹에 속하는지 저장
            data['cluster'] = model.fit_predict(data_scale)+1

            da = da.reset_index(drop = True)

            
            c = data[['place_id','place_longitude','place_latitude','cluster']]


            return c
    
    kmc =  k_means_clust(d,data3) 


    def cluster_list(kk):
        l = {}

        for i in range(d):
            l[i+1] = []

        for n in range(len(kk)):
            for m in range(1,d+1):
                if kk.iloc[n]['cluster'] ==m :
                    l[m].append(int(kk.iloc[n]['place_id']))

        return l


    cl = cluster_list(kmc)

    return cl
    

cluster(8,pppp)



#place_id
def data_x_y(num):
    m = pd.read_csv('seoul_result_db3.csv')
    sample = []
    id = []
    
    
    for j in num:
                sample.append((m.iloc[j]['place_longitude'] , m.iloc[j]['place_latitude'] ))
                id.append(m.iloc[j]['place_id'])

    return { '샘플' :sample , 
            'ID' : id
    }
data11 = data_x_y(num2)

#일정내 장소 위치
#p_pts = [(127, 37),(128, 37), (9, 7), (12, 9), (9,8) ,(21, 20), (13, 33), (7, 36) ]
p_pts = [(127, 37)]
ppp = {
    1: p_pts , 
    2: data11['샘플'] , 
    3: data11['ID']
}





from functools import reduce
import numpy as np
import pandas as pd

#place_id

def PossiblePlace(input):
    day_points = input[1]
    point = input[2]
    id  = input[3]
    ran =12

    def in_out(day_points ,point,id):
        dat = day_points 

        
        poly_list = []
        int1 = int(ran)
        
        def poly_data(dat):
            polydata = []
            
            
            for i in range((len(dat)-1)):

                int1 = int(ran)

                x = dat[i+1][0]-dat[i][0]
                y = dat[i+1][1]-dat[i][1]
                if y == 0:
                    y = 0.1


                rev_slope = -(x/y)

                x1 = dat[i][0]
                x2 = dat[i+1][0]

                y1 = dat[i][1]
                y2 = dat[i+1][1]

                fin_x1_1 = x1 + int1
                fin_x1_2 = x1 - int1
                
            
                fin_x2_1 = x2 + int1
                fin_x2_2 = x2 - int1
                

                equation1_1 = rev_slope*int1 + y1
                equation1_2 = rev_slope*-int1 + y1
                equation2_1 = rev_slope*int1 + y2
                equation2_2 = rev_slope*-int1 + y2

                points = [(fin_x1_1,equation1_1),(fin_x1_2,equation1_2),(fin_x2_1,equation2_1),(fin_x2_2,equation2_2)]


                polydata.append(points)
            

            return polydata
        p  =poly_data(dat)

        def convex_hull_graham(poly_points , point):
            TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

            def cmp(a, b):
                return float(a > b) - float(a < b)

            def turn(p, q, r):
                return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

            def _keep_left(hull, r):
                while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
                    
                    hull.pop()
                if not len(hull) or hull[-1] != r:
                    hull.append(r)
                return hull

            poly_points = sorted(poly_points)
            l = reduce(_keep_left, poly_points, [])
            u = reduce(_keep_left, reversed(poly_points), [])
            polygon = l.extend(u[i] for i in range(1, len(u) - 1)) or l


            inside = []
            inside_id = []
            in_index = []
            outside = []
            

            for j in range(len(point)) :
                N = len(polygon)-1    # N각형을 의미
                counter = 0
                p1 = polygon[0]
                for i in range(1, N+1):
                    p2 = polygon[i%N]
                    if point[j][1] > min(p1[1], p2[1]) and point[j][1] <= max(p1[1], p2[1]) and point[j][0] <= max(p1[0], p2[0]) and p1[1] != p2[1]:
                        xinters = (point[j][1]-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1]) + p1[0]
                        if(p1[0]==p2[0] or point[j][0]<=xinters):
                            counter += 1
                    p1 = p2 
                if counter % 2 == 0:

                    outside.append(point[j])

                else:
                    in_index.append(j)
                    inside.append(point[j])
                    inside_id.append(id[j])


            return inside_id
        
        for i in range(len(p)):
            poly_list.append(convex_hull_graham(p[i],point))

        return poly_list

    def num1_in_out(day_points,point,id):

        dat = day_points 

        

        poly_list = []
        
        def poly_data(dat):
            polydata = []
            
            
            for i in range((len(dat))):

                int1 = int(ran)

                x = dat[i][0]
                y = dat[i][1]

                slope = 1
                revers_slope = -(1/slope)

                point1 = (x , y+int1)
                point2 = (x , y-int1)
                point3 = (x+int1 , y)
                point4 = (x-int1 , y)
                point5 = (x+(int1*(2/3)) , y+(int1*(2/3)))
                point6 = (x+(int1*(2/3)) , y-(int1*(2/3)))
                point7 = (x-(int1*(2/3)) , y+(int1*(2/3)))
                point8 = (x-(int1*(2/3)) , y-(int1*(2/3)))
                
                

                points = [point1,point2,point3,point4,point5,point6,point7,point8]

                polydata.append(points)


            return polydata
        p  =poly_data(dat)

        def convex_hull_graham(poly_points , point):
            TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

            def cmp(a, b):
                return float(a > b) - float(a < b)

            def turn(p, q, r):
                return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

            def _keep_left(hull, r):
                while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
                    
                    hull.pop()
                if not len(hull) or hull[-1] != r:
                    hull.append(r)
                return hull

            poly_points = sorted(poly_points)
            l = reduce(_keep_left, poly_points, [])
            u = reduce(_keep_left, reversed(poly_points), [])
            polygon = l.extend(u[i] for i in range(1, len(u) - 1)) or l


            inside = []
            in_index = []
            outside = []
            inside_id = []
            

            for j in range(len(point)) :
                N = len(polygon)-1    # N각형을 의미
                counter = 0
                p1 = polygon[0]
                for i in range(1, N+1):
                    p2 = polygon[i%N]
                    if point[j][1] > min(p1[1], p2[1]) and point[j][1] <= max(p1[1], p2[1]) and point[j][0] <= max(p1[0], p2[0]) and p1[1] != p2[1]:
                        xinters = (point[j][1]-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1]) + p1[0]
                        if(p1[0]==p2[0] or point[j][0]<=xinters):
                            counter += 1
                    p1 = p2 
                if counter % 2 == 0:

                    outside.append(point[j])

                else:
                    in_index.append(j)
                    inside.append(point[j])
                    inside_id.append(id[j])

            return inside_id
        
        for i in range(len(p)):
            poly_list.append(convex_hull_graham(p[i],point))

        return poly_list

    if len(day_points) == 1:
        result = num1_in_out(day_points,point,id)[0]
    elif len(day_points) != 1:
        result = in_out(day_points,point,id)[0]

    if len(result) < 8 :
        result1 = num1_in_out(day_points,point,id)[0]
        
        for i in range(len(result1)):
            result.append(result1[i])

        result = set(tuple(result))
        result = list(result)
        

    return result


PossiblePlace(ppp)



def GetPlace(input):
    day_points = input[1]
    point = input[2]
    id  = input[3]

    id1 = id.copy()

    distance_list = []

    for i in range(len(point)):
        distance = abs((day_points[0][0]*10000 - point[i][0]*10000)  + abs( day_points[0][1]*10000 - point[i][1]*10000))
        distance_list.append(distance)
    '''   
    min_index =distance_list.index(min(distance_list))

    id_min =  id[min_index]

    '''
    id_list = []

    for i in range(len(id)):
        min_index =distance_list.index(min(distance_list))
        id_min =  id1[min_index]
        dis_min = distance_list[min_index]
        id_list.append(id_min)
        distance_list.remove(dis_min)
        id1.remove(id_min)

    

    return id_list


GetPlace(ppp)     

def TwoGetPlace(input):
    day_points = input[1]
    point = input[2]
    id  = input[3]
    id1 = id.copy()

    distance_list1 = []
    distance_list2 = []
    distance_list_All = []

    for i in range(len(point)):
        distance1 = abs((day_points[0][0]*10000 - point[i][0]*10000))  + abs((day_points[0][1]*10000 - point[i][1]*10000)) 
        distance2 = abs((day_points[1][0]*10000 - point[i][0]*10000))  + abs((day_points[1][1]*10000 - point[i][1]*10000))
        distance_list1.append(distance1)
        distance_list2.append(distance2)
        distance_list_All.append(distance1+distance2)
    '''
    min_index =distance_list_All.index(min(distance_list_All))

    id_min =  id[min_index]

    '''
    id_list = []

    for i in range(len(id)):
        min_index =distance_list_All.index(min(distance_list_All))
        id_min =  id1[min_index]
        dis_min = distance_list_All[min_index]
        id_list.append(id_min)
        distance_list_All.remove(dis_min)
        id1.remove(id_min)

    

    return id_list


TwoGetPlace(ppp)   






data = {
    1 : (127,37),
    2 : (127,38),
    3 : 25,
    4 : 10
}

def APIrout(input):
    start = input[1]
    end = input[2]
    walk = input[3]
    cycle = input[4]


    def get_directions(api_key, origin, destination):
        url = "https://apis-navi.kakaomobility.com/v1/directions"
        
        # 파라미터 추가
        params = {
            "origin": origin,
            "destination": destination
        }
        
        headers = {
            "Authorization": f"KakaoAK {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return None
        

    def direcrion(S_x , S_y , G_x , G_y)  :
        if __name__ == "__main__":
            REST_API_KEY = "5a85042810b38e43494af1c4c94b675e"
        
            origin_input = S_x , S_y 
            destination_input = G_x ,G_y
        
            result = get_directions(REST_API_KEY, origin_input, destination_input)
            if result:
                print(result['routes'][0]['summary']['duration']) #(json.dumps(result, indent=4))
                print(result)
            else:
                print("API 호출에 실패했습니다.")


    car  = direcrion(start[0] , start[1] , end[0] , end[1]) 


    #대중교통 api
    def Public(S_x , S_y , G_x , G_y):
        SX = S_x
        SY = S_y
        EX = G_x 
        EY =  G_y

        api_key = 'FoOUn1mJ5m+a/wEHhRH7LQ'

        url = f'https://api.odsay.com/v1/api/searchPubTransPathR?lang=0&SX={SX}&SY={SY}&EX={EX}&EY={EY}&apiKey={api_key}'
        response = requests.get(url).text
        response = json.loads(response)

        return response


    p = Public(start[0] , start[1] , end[0] , end[1])

    vec = ['도보' ,'자전거' , '대중교통','자차']
    pol = [0,0,50,100]

    #임의 설정

    w = 20
    cy = 15
    pub = 10 
    own_Car = 10

    

    v = vec[2]
    t = pub
    ww = w
    cc = cy
    p = pub*pol[2]

    if walk >= w : 
        v = vec[0]
        ww = walk-w
        p = pol[0]*w
        t = w
    else:

        if cycle >= cy:
            v = vec[1]
            cc = cycle - cy
            p = pol[1]*cycle
            t = cy
        else:
            pass 

    return{
        '교통수단' :v ,
        '소요시간' :  t ,
        '남은 도보' : ww,
        '자전거' : cc,
        '탄소배출량' : p
        }



APIrout(data)


def TSPRoute(input):
    p = len(input[3])
    num1 = input[3]

    #변경

    def data(input):
        

        sample_x = []
        sample_y = []

        num = input[3]

        '''
        for  i in l:
            a = Place.objects.get(pk=i).filter(place_latitude)
            b = Place.objects.get(pk=i).filter(place_longtitude)
            sample_x.append(a)
            sample_y.append(b)
        '''
        
        for j in range(len(num)):
                sample_x.append(input[2][j][0]*10000)
                sample_y.append(input[2][j][1]*10000 )

        #1자
        sample_dist = []
        #리스트 속 리스트.
        sample_dist_2 = []

        for k in range(len(sample_x)):
            for l in range(len(sample_y)):
                if k == l :
                    sample_dist.append(9999999999)

                elif k != l:
                    sample_dist.append(abs(sample_x[k]-sample_x[l] + sample_y[k]-sample_y[l]))


        for k in range(len(sample_x)):
            data = []
            for l in range(len(sample_y)):
            
                if k == l :
                    data.append(0)

                elif k != l:
                    data.append(abs(sample_x[k]-sample_x[l] + sample_y[k]-sample_y[l]))

            sample_dist_2.append(data)

        return {'li_in_li': sample_dist_2 ,
                'one' : sample_dist
                }
    
    data11 = data(input)['li_in_li'][0:p]
    """
    TSP 
    """
    def rout(data1):

        node = []

        def create_data_model(s):
            
        
            data = {}
            data["distance_matrix"] = s
            data["num_vehicles"] = 1
            data["depot"] = 0
            return data


        def print_solution(manager, routing, solution):
        
        
            #print(f"Objective: {solution.ObjectiveValue()} ")
            index = routing.Start(0)
            plan_output = "Route for vehicle 0:\n"
            route_distance = 0
            while not routing.IsEnd(index):
                node.append(manager.IndexToNode(index))
                plan_output += f" {manager.IndexToNode(index)} ->"
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

        
            plan_output += f" {manager.IndexToNode(index)}\n"
            #print(plan_output)
        
            plan_output += f"Route distance: {route_distance}\n"




        # 실행
        data = create_data_model(data1)

        # 루트 매니저
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        routing = pywrapcp.RoutingModel(manager)


        def distance_callback(from_index, to_index):
            
            # 노드 콜백
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        
        solution = routing.SolveWithParameters(search_parameters)

    
        if solution:
            print_solution(manager, routing, solution)
        
        node.append(0)

        
        return node
    # -> 데이터

    d = rout(data11)[0:-1]

    d_id = []
    
    for i in d:
        d_id.append(num1[i])


    return d_id



rout = TSPRoute(ppp)
rout



#place_id
num3 = [0,3,1,5,7,9]

#--> api 호출
#       도 자 버 지 차
val2 = [ 
        13,7,12,10,9,
        20,11,17,13,10,
        15,8,10,13,10,
        24,14,12,14,12,
        21,12,15,20,12 ,
        31,16,18,14,10
            ]



    """
    이동수단 선택 알고리즘
    """
    #        도 자 버 지 차
    def vec_optimizer(val1,p,c,w,eco):

        solver = pywraplp.Solver('Divorce Problem',
        pywraplp.Solver.FIXED_VALUE)

        vec = ['도보' ,'자전거' , '버스' , '지하철','자차']
        pol = [0,0,105,41,96]*p

        v =5

        val3 = []

        for i in range(len(val1)):
            val3.append(pol[i]*val1[i])
        
        #장소수 *변경
        place = [i for i in range(0,p)]   
        #자전거 제한   *변경
        cyl = c
        #도보 제한     *변경
        walk = w

        #이동수단별 시간  **자전거 불가시 9999 or inf**    *변경
        val = {'x_i' :val3}

        val4 = {'x_i' :val1}
        #전체 갯수
        n = [i for i in range(0,len(val['x_i']))]   

        #루트 변수
        d_1_i = {
            i : solver.IntVar(0,1,f"d_1{i}")
            for i in n 
        }

        solver.Add(sum([d_1_i[i]   for i in n ])  == len(place))

        #행 제약식
        for i in range(len(place)):
            solver.Add(sum([d_1_i[j+v*i] for j in range(v)]) ==1)
        

        # 최대 할당가능 제약식
        solver.Add(sum([ d_1_i[v*i]*val4['x_i'][v*i] for i in place ])<= walk )
        solver.Add(sum([ d_1_i[v*i+1]*val4['x_i'][v*i+1] for i in place ])<= cyl)
        
            
        
        #목적함수(합 최대화)
        if eco == 0 :
            solver.Minimize(sum([d_1_i[v*i+4]*val['x_i'][v*i+4] +d_1_i[v*i]*val['x_i'][v*i] + d_1_i[v*i+1]*val['x_i'][v*i+1] + d_1_i[v*i+3]*val['x_i'][v*i+3] + d_1_i[v*i+2]*val['x_i'][v*i+2]     for i in place]) )
        elif eco ==1 :
            solver.Maximize(sum([d_1_i[v*i+1]*val4['x_i'][v*i+1] + d_1_i[v*i]*val4['x_i'][v*i]     for i in place]) )
        #목적함수 (대중교통 최소화)
        #solver.Minimize(sum([d_1_i[5*i+3]*val['x_i'][5*i+3] + d_1_i[5*i+2]*val['x_i'][5*i+2]     for i in place]) )
        #목적함수 편차 최소화 
        #solver.Minimize()

        status = solver.Solve()

        result= {}


        for i in range(len(place)):
           result[i] = []



        if status == pywraplp.Solver.OPTIMAL:
            #print(f'An optimal solution was found!!')
            for i in n :
                if 0.01 <= d_1_i[i].solution_value() <=1 :
                    #print(f'{int(i//5)   ,  int(i%5)  } ==> {d_1_i[i].solution_value()}' ,"Value : ",val['x_i'][i] ,'    이동수단 :',vec[i%5])
                    #result[str(int(i//5))].append(int(i//5))
                    result[(int(i//v))].append(vec[i%v])
                    result[(int(i//v))].append(val4['x_i'][i])
                    result[(int(i//v))].append(val['x_i'][i])
                
            
            #print(f'Objective value = {solver.Objective().Value()}' )
        elif status == pywraplp.Solver.FEASIBLE:
            print(f'A feasible solution was found!!')
        elif status == pywraplp.Solver.INFEASIBLE:
            print(f'Infeasible!!')
        elif status == pywraplp.Solver.UNBOUNDED:
            print(f'Unbounded!!')
        else:
            print(f'Something went wrong... ResultStatus={status}')

        return result
    #eco 0 = 최단거리 , 1 = 탄소배출량 절감.
    a = vec_optimizer(val1,p,c,w,eco)
    
    return {'루트' : 
            d,
            '상세' : 
            a
            }

#        iloc 대중교통

TSPRoute(num3,val2,30,30)
