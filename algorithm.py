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

num2 = [0,24,35,48,150,220,120,640,221,525,555,93,110,130]

def clust(d,num1):
    #변경
    df2 = pd.read_csv('seoul_result_db3.csv')

    def data(m, n):
        

        sample_x = []
        sample_y = []

        num = n

        '''
        for  i in l:
            a = Place.objects.get(pk=i).filter(place_latitude)
            b = Place.objects.get(pk=i).filter(place_longtitude)
            sample_x.append(a)
            sample_y.append(b)
        '''
        
        for j in num:
                sample_x.append(m.iloc[j]['place_longitude'])
                sample_y.append(m.iloc[j]['place_latitude'] )

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
    data1 = data(df2,num1)['li_in_li']
    '''
    DB SCAN
    '''
    data3 = df2[['place_id','place_longitude', 'place_latitude']].iloc[num1]
    def db_scan(data):
        # 정규화 진행
        scaler = StandardScaler()
        df_scale = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)



        # epsilon, 최소 샘플 개수 설정
        model = DBSCAN(eps=1.0, min_samples=2)

        # 군집화 모델 학습 및 클러스터 예측 결과 반환
        model.fit(df_scale)
        df_scale['cluster'] = model.fit_predict(df_scale)




        plt.figure(figsize = (8, 8))

        # 이상치 번호는 -1, 클러스터 최대 숫자까지 iteration
        '''

        for i in range(-1, df_scale['cluster'].max() + 1):
            plt.scatter(df_scale.loc[df_scale['cluster'] == i, 'mapx'], df_scale.loc[df_scale['cluster'] == i, 'mapy'], 
                        label = 'cluster ' + str(i))

        '''

        f, ax = plt.subplots(2, 2)
        f.set_size_inches((12, 12))

        for i in range(4):
            # epsilon을 증가시키면서 반복
            eps = 0.4 + 0.2 * (i + 1)
            min_samples = 2
            

            # 군집화 및 시각화 과정 자동화
            model = DBSCAN(eps=eps, min_samples=min_samples)

            model.fit(df_scale)
            df_scale['cluster'] = model.fit_predict(df_scale)

            for j in range(-1, df_scale['cluster'].max() + 1):
                ax[i // 2, i % 2].scatter(df_scale.loc[df_scale['cluster'] == j, 'place_longitude'], df_scale.loc[df_scale['cluster'] == j, 'place_latitude'], 
                                label = 'cluster ' + str(j))

            ax[i // 2, i % 2].legend()
            ax[i // 2, i % 2].set_title('eps = %.1f, min_samples = %d'%(eps, min_samples), size = 15)
            ax[i // 2, i % 2].set_xlabel('Annual Income', size = 12)
            ax[i // 2, i % 2].set_ylabel('Spending Score', size = 12)
        

        return df_scale
    #dd = db_scan(data3)
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

            plt.figure(figsize = (8, 8))
            '''
            for i in range(k):
                    plt.scatter(data.loc[data['cluster'] == i+1, 'place_longitude'], data.loc[data['cluster'] == i+1, 'place_latitude'], 
                            label = 'cluster ' + str(i+1))

            plt.legend()
            plt.title('K = %d results'%k , size = 15)
            plt.xlabel('Annual Income', size = 12)
            plt.ylabel('Spending Score', size = 12)
            plt.show()
            '''
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
    

clust(8,num2)


num2 = [0,3,1,5,7,9,224,35,48,150,220,120,640,221,525,555,93,110,130]

#--> api 호출
val2 = [ 
        13,7,12,10,9,
        20,11,17,13,10,
        15,8,10,13,10,
        24,14,12,14,12,
        21,12,15,20,12 ,
        31,16,18,14,10
            ]

def rout_all(num1,val1,p,c,w,eco):
    #변경
    df2 = pd.read_csv('seoul_result_db3.csv')

    def data(m, n):
        

        sample_x = []
        sample_y = []

        num = n

        '''
        for  i in l:
            a = Place.objects.get(pk=i).filter(place_latitude)
            b = Place.objects.get(pk=i).filter(place_longtitude)
            sample_x.append(a)
            sample_y.append(b)
        '''
        
        for j in num:
                sample_x.append(m.iloc[j]['place_longitude']*10000 )
                sample_y.append(m.iloc[j]['place_latitude']*10000  )

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
    data1 = data(df2,num1)['li_in_li'][0:p]
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
        
        
            print(f"Objective: {solution.ObjectiveValue()} ")
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
            print(plan_output)
        
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
    d = rout(data1)

    """
    이동수단 선택 알고리즘
    """
    #        도 자 버 지 차
    def vec_optimizer(val1,p,c,w,eco):

        solver = pywraplp.Solver('Divorce Problem',
        pywraplp.Solver.FIXED_VALUE)

        vec = ['도보' ,'자전거' , '버스' , '지하철','자차']
        pol = [0,0,105,41,96]*p

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
            solver.Add(sum([d_1_i[j+5*i] for j in range(5)]) ==1)
        

        # 최대 할당가능 제약식
        solver.Add(sum([ d_1_i[5*i]*val4['x_i'][5*i] for i in place ])<= walk )
        solver.Add(sum([ d_1_i[5*i+1]*val4['x_i'][5*i+1] for i in place ])<= cyl)
        
            
        
        #목적함수(합 최대화)
        if eco == 0 :
            solver.Minimize(sum([d_1_i[5*i+4]*val['x_i'][5*i+4] +d_1_i[5*i]*val['x_i'][5*i] + d_1_i[5*i+1]*val['x_i'][5*i+1] + d_1_i[5*i+3]*val['x_i'][5*i+3] + d_1_i[5*i+2]*val['x_i'][5*i+2]     for i in place]) )
        elif eco ==1 :
            solver.Maximize(sum([d_1_i[5*i+1]*val4['x_i'][5*i+1] + d_1_i[5*i]*val4['x_i'][5*i]     for i in place]) )
        #목적함수 (대중교통 최소화)
        #solver.Minimize(sum([d_1_i[5*i+3]*val['x_i'][5*i+3] + d_1_i[5*i+2]*val['x_i'][5*i+2]     for i in place]) )
        #목적함수 편차 최소화 
        #solver.Minimize()


        status = solver.Solve()
        '''
        result = {
            '도보' : [],
            '자전거' : [],
            '버스' : [],
            '지하철' : [],
            '자차' : []
        }
        '''
        result= {}


        for i in range(len(place)):
           result[f'{i}'] = []



        if status == pywraplp.Solver.OPTIMAL:
            #print(f'An optimal solution was found!!')
            for i in n :
                if 0.01 <= d_1_i[i].solution_value() <=1 :
                    #print(f'{int(i//5)   ,  int(i%5)  } ==> {d_1_i[i].solution_value()}' ,"Value : ",val['x_i'][i] ,'    이동수단 :',vec[i%5])
                    result[str(int(i//5))].append(int(i//5))
                    result[str(int(i//5))].append(vec[i%5])
                    result[str(int(i//5))].append(val4['x_i'][i])
                    result[str(int(i//5))].append(val['x_i'][i])
                    '''
                    if  int(i%5) ==0: 
                        result['도보'].append([i//5,val['x_i'][i]])
                    elif  int(i%5) ==1: 
                        result['자전거'].append([i//5,val['x_i'][i]])
                    elif  int(i%5) ==2: 
                        result['버스'].append([i//5,val['x_i'][i]])
                    elif  int(i%5) ==3: 
                        result['지하철'].append([i//5,val['x_i'][i]])
                    elif  int(i%5) ==4: 
                        result['자차'].append([i//5,val['x_i'][i]])
                    '''
                
            
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
rout_all(num2,val2,6,20,20,1)

num2 = [0,24,35,48,150,220,120,640,221,525,555,93,110,130,1,3]

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

from functools import reduce
import numpy as np
p_pts = [(4, 20), (9, 7), (12, 9), (9,8) ,(21, 20), (13, 33), (7, 36) ]

points = [(9,10),(39,39),(9,39) , (9,11),(1,5),(33,12) ,(7,20),(20,10),(25,20),(15,30)]

#points = [ (126,37),(99,35),(128.7384361,34.8799083)]

def in_out(dat ,point,ran): 
    poly_list = []
    int1 = int(ran)
    
    def poly_data(dat):
        polydata = []
        
        
        for i in range((len(dat)-1)):

            int1 = int(ran)

            x = dat[i+1][0]-dat[i][0]
            y = dat[i+1][1]-dat[i][1]


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

        return {'polygon':polygon,
                'inside':inside , 
                'inside_index': in_index,
                'outside':outside}
    
    for i in range(len(p)):
        poly_list.append(convex_hull_graham(p[i],point))

    return poly_list

in_out(p_pts,points,9)


# 클러스터 한개일때 
def num1_in_out(dat,point,ran):

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

        return {'polygon':polygon,
                'inside':inside , 
                'inside_index': in_index,
                'outside':outside}
    
    for i in range(len(p)):
        poly_list.append(convex_hull_graham(p[i],point))

    return poly_list

num1_in_out(p_pts,p_pts,12)  