import pandas as pd
from haversine import haversine
import random
import copy
import math
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re

pd.set_option('display.max_columns', None)  # 设置显示所有列

# 两点时间计算
def time_compute(position1, position2, car_rare):
    distance = haversine(position1, position2)
    time = distance/car_rare
    return time

# 查看聚类的绘图（目前只设定了16种颜色，不能超16类
def plt_painting(coordinates_class):
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000','#0000FF', '#FF1493', '#800080', '#FF6347', '#00FFFF',
              '#FF69B4', '#CD5C5C', '#00FF00', '#FFD700', '#800000', '#8A2BE2', '#808080']
    plt.figure(figsize=(8, 6))
    for key, value in coordinates_class.items():
        plt.scatter(key[0], key[1], color=colors[value])
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Plot of Coordinate Points')
    plt.show()

# 根据经纬度确定分类
def update_category(row):
    key = (row['纬度'], row['经度'])
    if key in coordinates_class:
        return coordinates_class[key]
    return row['分类']

# 针对vip和重点客户的汽车挑选
def customer_car_choice(customer_info, volume_dict, float_rare, car_time_table, car_schedule_table, depot, car_rare):
    need_volume = customer_info[4]
    useful_car_volume = []
    choice_car = -1
    for key in volume_dict.keys():
        if key * float_rare >= need_volume:
            useful_car_volume.append(key)
    if useful_car_volume == []:
        useful_car_volume.append(max(volume_dict.keys()))
    useful_car_volume_copy = copy.deepcopy(useful_car_volume)
    choice_car_volume = random.choice(useful_car_volume_copy)
    choice_car_list = volume_dict[choice_car_volume]
    choice_car_list_copy = copy.deepcopy(choice_car_list)
    while choice_car == -1:
        pre_choice_car = random.choice(choice_car_list_copy)
        if not car_time_table[pre_choice_car]:
            choice_car = pre_choice_car
        else:
            end_time = car_time_table[pre_choice_car][-1][-1][1]
            end_coordinate = car_schedule_table[pre_choice_car][-1][-1]
            time1 = time_compute([end_coordinate[0],end_coordinate[1]], depot, car_rare)
            time2 = time_compute(depot, [customer_info[0],customer_info[1]], car_rare)
            if end_time + time1 + time2 <= time_dict[customer_info[2]][1]:
                choice_car = pre_choice_car
            else:
                choice_car_list_copy.remove(pre_choice_car)
                if not choice_car_list_copy:
                    useful_car_volume_copy.remove(choice_car_volume)
                    if not useful_car_volume_copy:
                        next_choice_volume = [x for x in volume_dict.keys() if x not in useful_car_volume]
                        choice_car_volume = max(next_choice_volume)
                        choice_car_list = volume_dict[choice_car_volume]
                        choice_car_list_copy = copy.deepcopy(choice_car_list)
                    else:
                        choice_car_volume = random.choice(useful_car_volume_copy)
                        choice_car_list = volume_dict[choice_car_volume]
                        choice_car_list_copy = copy.deepcopy(choice_car_list)
    # 返回车牌号
    return choice_car

# 一辆车记录和更新数据
def car_update_info(car, route, start_time):
    distance = 0
    car_time_table[car].append([])
    car_schedule_table[car].append([])
    start = depot
    for coordinate in route:
        coordinate_schedule_table[coordinate[2]][coordinate[0],coordinate[1]] = 1
        need_time = time_compute(start, [coordinate[0],coordinate[1]], car_rare)
        distance += need_time * car_rare
        for x in area_order_dict[coordinates_class[coordinate[0],coordinate[1]]][coordinate[0],coordinate[1]][coordinate[2]]:
            order_schedule[x[0]] = [car, car_num_dict[car], start_time, start_time + need_time]
        car_time_table[car][car_num_dict[car]].append([start_time, start_time+need_time])
        car_schedule_table[car][car_num_dict[car]].append(coordinate)
        start = [coordinate[0],coordinate[1]]
        start_time += need_time
    need_time = time_compute([route[-1][0],route[-1][1]], depot, car_rare)
    distance += need_time * car_rare
    car_time_table[car][car_num_dict[car]].append([start_time, start_time + need_time])
    car_schedule_table[car][car_num_dict[car]].append([depot[0],depot[1],-1])
    car_num_dict[car] += 1
    return distance

# 扩展时间窗的路径扩展和安排
def car_route_scheduling(customer_info, car, start_time, important_volume_list):
    remain_volume = car_dict[car] * float_rare - customer_info[4]
    route = [[customer_info[0],customer_info[1],customer_info[2]]]
    if remain_volume <= 0:
        distance = car_update_info(car, route, start_time)
        load = customer_info[4]
    else:
        more_list = []
        for item in important_volume_list:
            if coordinate_schedule_table[item[2]][item[0], item[1]] == -1:
                time = time_compute([customer_info[0], customer_info[1]], [item[0],item[1]], car_rare)
                # 纬度 经度 公司级别 体积 时间 分类
                more_list.append([item[0], item[1], item[2], coordinate_volume[item[0], item[1]][item[2]], time, item[3]])
        more_list = sorted(more_list, key=lambda x: x[4], reverse=True)
        more_list = [x for x in more_list if x[3] <= remain_volume]
        vip_list = [x for x in more_list if x[2] == 5]
        vip_list_copy = copy.deepcopy(vip_list)
        key_list = [x for x in more_list if x[2] == 3]
        key_list_copy = copy.deepcopy(key_list)
        time = time_compute(depot, [customer_info[0], customer_info[1]],  car_rare)
        end_time = start_time + time
        label = customer_info[3]
        while vip_list_copy:
            for item in vip_list:
                if remain_volume >= item[3]:
                    need_time = time_compute([route[-1][0],route[-1][1]], [item[0], item[1]],  car_rare)
                    if end_time + need_time <= time_dict[5][1]:
                        route.append([item[0],item[1],item[2]])
                        remain_volume -= item[3]
                        end_time += need_time
                        label = item[5]
                        coordinate_schedule_table[item[2]][item[0], item[1]] = 1
                        vip_list_copy.remove(item)
                    else:
                        vip_list_copy.remove(item)
                else:
                    vip_list_copy.remove(item)
        while key_list_copy:
            for item in key_list:
                if remain_volume >= item[3]:
                    need_time = time_compute([route[-1][0],route[-1][1]], [item[0], item[1]],  car_rare)
                    if end_time + need_time <= time_dict[3][1]:
                        route.append([item[0],item[1],item[2]])
                        remain_volume -= item[3]
                        end_time += need_time
                        label = item[5]
                        coordinate_schedule_table[item[2]][item[0], item[1]] = 1
                        key_list_copy.remove(item)
                    else:
                        key_list_copy.remove(item)
                else:
                    key_list_copy.remove(item)
        if remain_volume >= 0:
            time_list = []
            for item in regular_list:
                if coordinate_schedule_table[item[2]][item[0],item[1]] == -1 and (item[0],item[1]) in area_order_dict[label].keys():
                    time = time_compute([customer_info[0], customer_info[1]], [item[0],item[1]], car_rare)
                    time_list.append([item[0], item[1],item[2],coordinate_volume[item[0], item[1]][item[2]], time])
            time_list = sorted(time_list, key=lambda x: x[3], reverse=True)
            for item in time_list:
                if remain_volume >= item[3]:
                    route.append([item[0], item[1],item[2]])
                    remain_volume -= item[3]
                    if remain_volume == 0:
                        break
        distance = car_update_info(car, route, start_time)
        load = car_dict[car] * float_rare - remain_volume
    return load, distance

# 输出文件
def output_file():
    output_data = []
    for key, value in order_schedule.items():
        if value != -1:
            # ERP单号 车牌 序列 到达时间 纬度 经度度 目的地址 公司级别
            result = order_df[order_df['ERP单号'] == key].values.tolist()[0]
            hours = int(value[3])
            minutes = math.floor((value[3] - hours) * 60)
            true_time = f"{hours}:{minutes:02d}"
            data = [key, value[0], value[1], true_time, result[2], result[3], result[6],result[4], result[1]]
            output_data.append(data)
    output_df = pd.DataFrame(output_data)
    output_df.columns = ['ERP单号', '车牌号', '序列', '到达时间', '纬度', '经度', '目的地址', '体积','公司级别']
    output_df['combined'] = output_df['车牌号'].astype(str) + '_' + output_df['序列'].astype(str)
    output_df = output_df.sort_values(by=['combined', '到达时间'])
    output_df = output_df.drop('combined', axis=1)
    output_df.to_excel('output.xlsx', index=False)

if __name__ == '__main__':
    # 参数设置
    k = 8  # 聚类数量
    float_rare = 1.01  # 容积浮动系数
    greedy_rare = 0.8  # 贪婪系数
    car_rare = 40  # 汽车速度
    time_dict = {5:[8,9],3:[9,10],0:[8,20]}  # 各等级客户时间窗

    # 初始化评价指标
    total_distance = 0  # 总距离
    load_list = []  # 平均负载率
    total_car_num = 0  # 车辆数

    # 数据处理
    # 读取订单数据
    all_order_df = pd.read_excel('订单信息.xlsx', skiprows=1)
    day_order_df = all_order_df[all_order_df['下单时间'].str.contains('11-07') & (all_order_df['业务属性'] == 'D') & ~all_order_df['目的地详细地址'].str.contains('绥德路99号')]
    day_order_df = day_order_df[day_order_df['经度'].notnull()]
    day_order_df['体积'] = day_order_df['体积'].fillna(0)
    day_order_df['分类'] = -1

    # 仓库位置 [纬度, 经度]
    depot = [31.264033, 121.378945]

    # 所有经纬度坐标
    unique_position = day_order_df[['纬度', '经度']].drop_duplicates()
    unique_position = unique_position.values.tolist()

    # 创建KMeans模型并进行训练
    coordinates = np.array(unique_position)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(coordinates)
    labels = kmeans.labels_
    coordinates_class = {}
    for i in range(len(unique_position)):
        coordinates_class[unique_position[i][0],unique_position[i][1]] = labels[i]

    # 更新每个订单所属的区域分类
    day_order_df['分类'] = day_order_df.apply(update_category, axis=1)

    # vip客户 重点客户 普通客户
    day_order_df.loc[day_order_df['公司级别'].isnull(), '公司级别'] = 0
    order_df = day_order_df[['ERP单号','公司级别', '纬度', '经度', '体积', '分类', '目的地详细地址']]
    order_list = order_df.values.tolist()

    vip_customer_df = order_df[order_df['公司级别'] == 5]
    key_customer_df = order_df[order_df['公司级别'] == 3]
    regular_customer_df = order_df[order_df['公司级别'] == 0]

    # 根据区域记录订单
    area_order_dict = {key: {} for key in range(k)}  # 由 [分类][经纬度][级别]为索引 ERP单号，体积
    area_total_volume = {key: 0 for key in range(k)}  # 分类订单总体积
    coordinate_volume = {key: {} for key in coordinates_class.keys()}  # 坐标点各级别客户总体积
    order_schedule = {}  # 记录ERP单号的分配车辆和路程的信息
    for order in order_list:
        area_total_volume[order[5]] += order[4]
        order_schedule[order[0]] = []
        if (order[2],order[3]) not in area_order_dict[order[5]].keys():
            area_order_dict[order[5]][order[2], order[3]] = {}
            area_order_dict[order[5]][order[2], order[3]][order[1]] = []
            area_order_dict[order[5]][order[2], order[3]][order[1]].append([order[0], order[4]])
        else:
            if order[1] not in area_order_dict[order[5]][order[2], order[3]].keys():
                area_order_dict[order[5]][order[2], order[3]][order[1]] = []
                area_order_dict[order[5]][order[2], order[3]][order[1]].append([order[0], order[4]])
            else:
                area_order_dict[order[5]][order[2], order[3]][order[1]].append([order[0], order[4]])
        if order[1] not in coordinate_volume[order[2], order[3]]:
            coordinate_volume[order[2], order[3]][order[1]] = order[4]
        else:
            coordinate_volume[order[2], order[3]][order[1]] += order[4]

    # 读取车辆数据
    car_df = pd.read_excel('车辆信息.xlsx')
    car_dict = {}
    volume_dict = {}
    for index, row in car_df.iterrows():
        car_dict[row['车牌号']] = row['容积（立方）']
        if row['容积（立方）'] not in volume_dict:
            volume_dict[row['容积（立方）']] = [row['车牌号']]
        else:
            volume_dict[row['容积（立方）']].append(row['车牌号'])

    # 所有类型顾客的信息列表
    customer_list = order_df.groupby(['纬度', '经度', '公司级别', '分类'])['体积'].sum().reset_index().values.tolist()
    customer_list = sorted(customer_list, key=lambda x: x[4], reverse=True)

    # vip客户和key客户的体积信息
    vip_volume_list = [x for x in customer_list if x[2] == 5]
    key_volume_list = [x for x in customer_list if x[2] == 3]
    regular_list = [x for x in customer_list if x[2] == 0]
    important_volume_list = [x for x in customer_list if x[2] != 0]

    # # # 算法流程
    # 将原问题划分为多个子问题，使用贪婪策略结合cvrptw优化,原则先保证vip和重点客户的派送，基于vip和重点客户点进行同区域贪婪的选择一条
    # 初始化车辆时间表
    car_time_table = {key: [] for key in car_dict.keys()}  # 记录车辆时间 [[[start, end],[start, end],...],[[start, end],...]...]
    car_schedule_table = {key: [] for key in car_dict.keys()}  # 记录车辆任务end的经纬度和客户类 [[[纬度,经度],[纬度,经度]....],[[纬度,经度],...]...]
    car_num_dict = {key: 0 for key in car_dict.keys()}  # 记录车辆有几趟任务
    coordinate_schedule_table = {key: {} for key in time_dict.keys()}
    for x in coordinate_schedule_table.keys():
        coordinate_schedule_table[x] = {key: -1 for key in coordinates_class.keys()}  # 记录每个等级的经纬度的客户是否已分配

    # 分配vip客户
    for vip_info in important_volume_list:
        # 如果该vip客户还未分配
        if coordinate_schedule_table[vip_info[2]][vip_info[0],vip_info[1]] == -1:
            # 查看附近的客户点
            car = customer_car_choice(vip_info, volume_dict, float_rare, car_time_table, car_schedule_table, depot, car_rare)
            # 如果所选车辆容积小于等于该客户的体积，则全给这辆车
            if car_dict[car] * float_rare <= vip_info[4]:
                if not car_time_table[car]:
                    end_time = 8
                    need_time = time_compute(depot, [vip_info[0], vip_info[1]], car_rare)
                    start_time = 8 - need_time
                    route = [[vip_info[0], vip_info[1], vip_info[2]]]
                    distance = car_update_info(car, route, start_time)
                    total_distance += distance
                    load_list.append(vip_info[4]/car_dict[car])
                else:
                    start_time = car_time_table[car][-1][-1][1]
                    route = [[vip_info[0], vip_info[1], vip_info[2]]]
                    distance = car_update_info(car, route, start_time)
                    total_distance += distance
                    load_list.append(vip_info[4] / car_dict[car])
            else:
                if not car_time_table[car]:
                    end_time = 8
                    need_time = time_compute(depot, [vip_info[0], vip_info[1]], car_rare)
                    start_time = 8 - need_time
                    load, distance = car_route_scheduling(vip_info, car, start_time, important_volume_list)
                    total_distance += distance
                    load_list.append(load/car_dict[car])
                else:
                    start_time = car_time_table[car][-1][-1][1]
                    load, distance = car_route_scheduling(vip_info, car, start_time, important_volume_list)
                    total_distance += distance
                    load_list.append(load/car_dict[car])

    # 计算各区域剩余体积
    for key, value in coordinate_schedule_table.items():
        for key1,value1 in coordinate_schedule_table[key].items():
            if value1 != -1:
                area_total_volume[coordinates_class[key1]] -= coordinate_volume[key1][key]

    regular_list_copy = [x for x in regular_list if coordinate_schedule_table[0][x[0], x[1]] == -1]
    area_order_dict_copy = copy.deepcopy(area_order_dict)

    # 各区域普通客户分配 ———— 优先安排体积最大的点，贪婪
    regular_coordinate_schedule = coordinate_schedule_table[0]
    area_total_volume = dict(sorted(area_total_volume.items(), key=lambda x: x[1]))
    choice_car_list = {key: [] for key in volume_dict.keys()}  # 各容积可选的车
    choice_volume_dict = {key: 0 for key in volume_dict.keys()}  # 可选的容积
    for key,value in volume_dict.items():
        for value1 in value:
            if not car_time_table[value1] or car_time_table[value1][-1][-1][1] <= (time_dict[0][1] - 2):
                choice_car_list[key].append(value1)
                choice_volume_dict[key] = 1
    area_total_distance = 0

    while regular_list_copy:
        for item in regular_list_copy:
            area_customer = [y for y in regular_list_copy if y[3] == item[3] and y != item]
            big_suit_volume = [i for i in choice_car_list.keys() if i * float_rare >= item[4] and i in choice_volume_dict.keys()]
            if not big_suit_volume:
                choice_volume = max(choice_volume_dict.keys())
            else:
                small_suit_volume = [i for i in big_suit_volume if i <= area_total_volume[item[3]] * 1.5 and i in choice_volume_dict.keys()]
                if not small_suit_volume:
                    choice_volume = min(choice_volume_dict.keys())
                else:
                    choice_volume = random.choice(small_suit_volume)
            busy_car = [car for car in choice_car_list[choice_volume] if car_time_table[car]]
            idly_car = [car for car in choice_car_list[choice_volume] if not car_time_table[car]]
            if busy_car:
                car = random.choice(busy_car)
            else:
                car = random.choice(idly_car)
            if not car_time_table[car]:
                start_time = 8 - time_compute(depot, [item[0],item[1]], car_rare)
                end_time = 8
            else:
                start_time = car_time_table[car][-1][-1][1]
                end_time = car_time_table[car][-1][-1][1] + time_compute(depot, [item[0],item[1]], car_rare)
            car_time_table[car].append([])
            car_schedule_table[car].append([])
            car_time_table[car][car_num_dict[car]].append([start_time, end_time])
            car_schedule_table[car][car_num_dict[car]].append([item[0],item[1],item[2]])
            area_total_distance += haversine(depot, [item[0],item[1]])
            # 如果车辆容积较大，扩展路径
            remain_volume = car_dict[car] * float_rare - item[4]
            if remain_volume > 0:
                for x in area_order_dict_copy[item[3]][item[0], item[1]][item[2]]:
                    order_schedule[x[0]] = [car, car_num_dict[car], start_time, end_time]
                area_order_dict_copy[item[3]][item[0], item[1]][item[2]] = []
                start_item = item
                while remain_volume > 0 and end_time < time_dict[0][1]:
                    time_list = []
                    for area_coordinate in area_customer:
                        if area_coordinate[4] <= remain_volume:
                            need_time = time_compute([start_item[0], start_item[1]], [area_coordinate[0], area_coordinate[1]], car_rare)
                            time_list.append([area_coordinate[0],area_coordinate[1],area_coordinate[2],area_coordinate[3],area_coordinate[4],need_time])
                    if time_list:
                        time_list = sorted(time_list, key=lambda x: x[5])
                        if end_time + time_compute([start_item[0],start_item[1]], [time_list[0][0],time_list[0][1]], car_rare) > time_dict[0][1]:
                            break
                        else:
                            area_total_distance += haversine([start_item[0], start_item[1]], [time_list[0][0],time_list[0][1]])
                            start_time = end_time
                            end_time += time_compute([start_item[0], start_item[1]], [time_list[0][0], time_list[0][1]], car_rare)
                            start_item = time_list[0]
                            remain_volume -= time_list[0][4]
                            area_customer.remove([time_list[0][0],time_list[0][1],time_list[0][2],time_list[0][3],time_list[0][4]])
                            regular_list_copy.remove([time_list[0][0],time_list[0][1],time_list[0][2],time_list[0][3],time_list[0][4]])
                            car_time_table[car][car_num_dict[car]].append([start_time, end_time])
                            car_schedule_table[car][car_num_dict[car]].append([time_list[0][0], time_list[0][1],time_list[0][2]])
                            for x in area_order_dict_copy[time_list[0][3]][time_list[0][0], time_list[0][1]][time_list[0][2]]:
                                order_schedule[x[0]] = [car, car_num_dict[car], start_time, end_time]
                            area_order_dict_copy[item[3]][item[0], item[1]][item[2]] = []
                    else:
                        break
                need_time = time_compute([start_item[0],start_item[1]], depot, car_rare)
                car_time_table[car][car_num_dict[car]].append([end_time, end_time+need_time])
                car_schedule_table[car][car_num_dict[car]].append([depot[0],depot[1],-1])
                area_total_distance += haversine([start_item[0], start_item[1]], depot)
                car_num_dict[car] += 1
                area_total_volume[item[3]] -= (car_dict[car] * float_rare - item[4] - remain_volume)
                load = (car_dict[car] * float_rare - item[4] - remain_volume)
                load_list.append(load/car_dict[car])
                if end_time + need_time >= (time_dict[0][1] - 2):
                    choice_car_list[choice_volume].remove(car)
                    if not choice_car_list[choice_volume]:
                        del choice_volume_dict[choice_volume]
                regular_list_copy.remove(item)
            # 如果车辆容积较小,还需要选择车辆参与
            else:
                index_to_change = None
                for index, sublist in enumerate(regular_list_copy):
                    if item == sublist:
                        index_to_change = index
                        break
                remain_volume = car_dict[car] * float_rare
                choice_order = area_order_dict_copy[item[3]][item[0], item[1]][item[2]]
                choice_order = sorted(choice_order, key=lambda x: x[1], reverse=True)
                for order_info in choice_order:
                    if remain_volume == 0:
                        break
                    if remain_volume > order_info[1]:
                        remain_volume -= order_info[1]
                        area_total_volume[item[3]] -= order_info[1]
                        order_schedule[order_info[0]] = [car, car_num_dict[car], start_time, end_time]
                        regular_list_copy[index_to_change][4] -= order_info[1]
                        area_order_dict_copy[item[3]][item[0], item[1]][item[2]].remove(order_info)
                need_time = time_compute(depot, [item[0],item[1]], car_rare)
                car_time_table[car][car_num_dict[car]].append([end_time, end_time + need_time])
                car_schedule_table[car][car_num_dict[car]].append([depot[0], depot[1], -1])
                area_total_distance += haversine([item[0],item[1]], depot)
                car_num_dict[car] += 1

    total_distance += area_total_distance
    # 输出文件
    output_file()
    # 各目标值
    for key,value in car_num_dict.items():
        if value != 0:
            total_car_num += 1
    print(f'总行驶距离为{total_distance}')
    print(f'平均负载率为{sum(load_list)/len(load_list)}')
    print(f'使用的车辆数为{total_car_num}')



