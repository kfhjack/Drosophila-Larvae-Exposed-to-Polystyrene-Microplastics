import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv


def dist(vecA, vecB):
    return np.sqrt(np.sum(np.power(np.array(vecA) - np.array(vecB), 2)))


def pretreatment(c):
    for x in range(1, fps_num):
        for y in range(fly_num):
            if np.isnan(locations[x, 0, 0, y]) or dist(locations[x, 0, :, y], locations[x - 1, 0, :, y]) > 100:
                locations[x, 0, :, y] = locations[x - 1, 0, :, y]
    for j in range(fly_num):
        temp = 0
        zero_normal_temp, zeroone_normal_temp, one_normal_temp, ten_normal_temp, twenty_normal_temp = [], [], [], [], []
        for u in range(fps_num):
            if temp == 0:
                if c == 0:
                    zero_normal_temp.append(dist(locations[u, 0, :, j], [550, 550]) / 450)
                if c == 1:
                    zeroone_normal_temp.append(dist(locations[u, 0, :, j], [550, 550]) / 450)
                if c == 2:
                    one_normal_temp.append(dist(locations[u, 0, :, j], [550, 550]) / 450)
                if c == 3:
                    ten_normal_temp.append(dist(locations[u, 0, :, j], [550, 550]) / 450)
                if c == 4:
                    twenty_normal_temp.append(dist(locations[u, 0, :, j], [550, 550]) / 450)
            if dist(locations[u, 0, :, j], [550, 550]) > 450:
                temp = 1
            if temp == 1:
                locations[u, 0, :, j] = np.nan
                if c == 0:
                    zero_normal_temp.append(1)
                if c == 1:
                    zeroone_normal_temp.append(1)
                if c == 2:
                    one_normal_temp.append(1)
                if c == 3:
                    ten_normal_temp.append(1)
                if c == 4:
                    twenty_normal_temp.append(1)
        if a == 0:
            zero_normal.append(zero_normal_temp)
        if a == 1:
            zeroone_normal.append(zeroone_normal_temp)
        if a == 2:
            one_normal.append(one_normal_temp)
        if a == 3:
            ten_normal.append(ten_normal_temp)
        if a == 4:
            twenty_normal.append(twenty_normal_temp)
    return locations


def track():
    r = 450.0
    ax, bx = (550, 550)
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = ax + r * np.cos(theta)
    y = bx + r * np.sin(theta)
    plt.plot(x, y)
    for i in range(date.shape[3]):
        plt.plot(date[:, 0, 0, i], date[:, 0, 1, i])
    plt.show()


def draw_speed_time():
    color = ["red", "blue", "black", "green", "yellow"]
    speed_plot = []
    for d in range(5):
        speed_ave = []
        for e in range(fps_num - 30):
            speed_sum = 0
            num = len(speed_tol[d])
            for g in range(len(speed_tol[d])):
                if e < len(speed_tol[d][g]) and speed_tol[d][g][e] > 0.1:
                    speed_sum += speed_tol[d][g][e]
                else:
                    num = num - 1
            if num != 0:
                speed_ave.append(speed_sum / num)
            else:
                speed_ave.append(0)
        speed_plot.append(speed_ave)
    x = range(0, fps_num - 30, 1)
    for u in range(5):
        y = speed_plot[u]
        plt.plot(x, y, color=color[u])
        plt.xlabel('fps', fontproperties='Times New Roman', fontsize=15, weight='bold')
        plt.ylabel('mm/s', fontproperties='Times New Roman', fontsize=15, weight='bold')
        plt.legend(["control", "0.1%", "1%", "10%", "20%"],
                   prop={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'}, loc=1)
    plt.show()


def draw_angle_time():
    color = ["red", "blue", "black", "green", "yellow"]
    angle_plot = []
    for d in range(5):
        angle_ave = []
        for e in range(fps_num - 30):
            angle_sum = 0
            num = len(angle_tol[d])
            for g in range(len(angle_tol[d])):
                if e < len(angle_tol[d][g]):
                    angle_sum += angle_tol[d][g][e]
                else:
                    num = num - 1
            angle_ave.append(angle_sum / num)
        angle_plot.append(angle_ave)
    x = range(0, fps_num - 30, 1)
    for u in range(5):
        y = angle_plot[u]
        plt.plot(x, y, color=color[u])
        plt.xlabel('fps', fontproperties='Times New Roman', fontsize=15, weight='bold')
        plt.ylabel('rad/s', fontproperties='Times New Roman', fontsize=15, weight='bold')
        plt.legend(["control", "0.1%", "1%", "10%", "20%"],
                   prop={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'}, loc=1)
    plt.show()


def draw_speed_in():
    temp = []
    for s in range(len(speed_tol)):
        alla = []
        for q in range(len(speed_tol[s])):
            speed_ave = 0
            speed_ave_tol = []
            for r in range(len(speed_tol[s][q])):
                speed_ave += speed_tol[s][q][r]
                if r % 30 == 0 and r != 0:
                    speed_ave_tol.append(speed_ave / 30)
                    speed_ave = 0
            alla.append(speed_ave_tol)
        temp.append(alla)
    return temp


def draw_angle_in():
    temp = []
    for s in range(len(angle_tol)):
        alla = []
        for q in range(len(angle_tol[s])):
            angle_ave = 0
            angle_ave_tol = []
            for r in range(len(angle_tol[s][q])):
                angle_ave += angle_tol[s][q][r]
                if r % 30 == 0 and r != 0:
                    angle_ave_tol.append(angle_ave / 30)
                    angle_ave = 0
            alla.append(angle_ave_tol)
        temp.append(alla)
    return temp


def draw_speed_odd():
    speed_odds = []
    for n in range(len(speed_tol)):
        speed_odd = []
        for m in range(len(speed_tol[n])):
            slow, middle, fast = 0, 0, 0
            for u in range(len(speed_tol[n][m])):
                if 0.1 < speed_tol[n][m][u] <= 0.4:
                    slow += 1
                if 0.4 < speed_tol[n][m][u] <= 0.7:
                    middle += 1
                if 0.7 < speed_tol[n][m][u] <= 1:
                    fast += 1
            speed_sum = slow + middle + fast
            speed_odd.append([slow / speed_sum, middle / speed_sum, fast / speed_sum])
        speed_odds.append(speed_odd)
    return speed_odds


def draw_angle_odd():
    angle_odds = []
    for n in range(len(angle_tol)):
        angle_odd = []
        for m in range(len(angle_tol[n])):
            slow, middle, fast = 0, 0, 0
            for u in range(len(angle_tol[n][m])):
                if 0 <= angle_tol[n][m][u] < 45:
                    slow += 1
                if 45 <= angle_tol[n][m][u] < 90:
                    middle += 1
                if 90 <= angle_tol[n][m][u] <= 180:
                    fast += 1
            angle_sum = slow + middle + fast
            angle_odd.append([slow / angle_sum, middle / angle_sum, fast / angle_sum])
        angle_odds.append(angle_odd)
    return angle_odds


def draw_speed_odds():
    color = ["red", "blue", "black", "green", "yellow"]
    speed_odds = []
    for n in range(len(speed_tol)):
        speed_odd = []
        speed_group = []
        for m in range(len(speed_tol[n])):
            for u in range(len(speed_tol[n][m])):
                speed_group.append(speed_tol[n][m][u])
        bins = np.arange(0, 1.5, 0.01)
        speed_cut = pd.cut(speed_group, bins)
        counts = pd.value_counts(speed_cut, sort=False)
        sum_speed = np.sum(counts.values)
        for t in range(len(counts.values)):
            speed_odd.append(counts.values[t] / sum_speed)
        speed_odds.append(speed_odd)
    x = np.arange(0, 1.49, 0.01)
    for g in range(5):
        y = speed_odds[g]
        plt.plot(x, y, color=color[g])
    plt.xlabel('mm/s', fontproperties='Times New Roman', fontsize=15, weight='bold')
    plt.ylabel('%', fontproperties='Times New Roman', fontsize=15, weight='bold')
    plt.legend(["control", "0.1%", "1%", "10%", "20%"],
               prop={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'}, loc=1)
    plt.show()


def draw_angle_odds(angle_tol_1):
    color = ["red", "blue", "black", "green", "yellow"]
    angle_odds = []
    for n in range(len(angle_tol_1)):
        angle_odd = []
        angle_group = []
        for m in range(len(angle_tol_1[n])):
            for u in range(len(angle_tol_1[n][m])):
                angle_group.append(angle_tol_1[n][m][u])
        bins = np.arange(0, 180, 1)
        angle_cut = pd.cut(angle_group, bins)
        counts = pd.value_counts(angle_cut, sort=False)
        sum_angle = np.sum(counts.values)
        for t in range(len(counts.values)):
            angle_odd.append(counts.values[t] / sum_angle)
        angle_odds.append(angle_odd)
    x = np.arange(0, 179, 1)
    for g in range(5):
        y = angle_odds[g]
        plt.plot(x, y, color=color[g])
    plt.xlabel('rad/s', fontproperties='Times New Roman', fontsize=15, weight='bold')
    plt.ylabel('%', fontproperties='Times New Roman', fontsize=15, weight='bold')
    plt.legend(["control", "0.1%", "1%", "10%", "20%"],
               prop={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'}, loc=1)

    plt.show()


def draw_speed():
    color = ["red", "blue", "black", "green", "yellow"]
    for c in range(5):
        bf = plt.boxplot(speed_tol[c], positions=[c], widths=0.5, patch_artist=True, showfliers=False)
        plt.setp(bf["boxes"], facecolor=color[c])
    plt.ylabel('s', fontproperties='Times New Roman', fontsize=15, weight='bold')
    x = range(0, 5, 1)
    plt.xticks(x, ["0", '0.5', "1", "10", "20"])
    plt.show()


def draw_len():
    color = ["red", "blue", "black", "green", "yellow"]
    for c in range(5):
        plt.scatter(time_tol[c], length_tol[c], c=color[c], marker='.', s=100)
    x = range(0, 350, 50)
    plt.xticks(x, ["0", "50", '100', "150", "200", "250", "end"])
    plt.xlabel('s', fontproperties='Times New Roman', fontsize=15, weight='bold')
    plt.ylabel('mm', fontproperties='Times New Roman', fontsize=15, weight='bold')
    plt.legend(["control", "0.1%", "1%", "10%", "20%"],
               prop={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'})
    for w in range(5):
        plt.scatter(np.mean(time_tol[w]), np.mean(length_tol[w]), c=color[w], marker='.', s=400)
    plt.show()


def draw_num():
    x = range(0, 5, 1)
    y = [len(time_tol[0]), len(time_tol[1]), len(time_tol[2]), len(time_tol[3]), len(time_tol[4])]
    plt.plot(x, y)
    plt.ylabel('number', fontproperties='Times New Roman', fontsize=15, weight='bold')
    plt.xticks(x, ["0", '0.5', "1", "10", "20"])
    plt.show()


def GetAngle(part1, part2, part3, part4):
    dx1 = part1[0] - part2[0]
    dy1 = part1[1] - part2[1]
    dx2 = part3[0] - part4[0]
    dy2 = part3[1] - part4[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180 / math.pi
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180 / math.pi
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    return insideAngle


speed_tol = []
angle_tol = []
time_tol = []
length_tol = []
angle_fre_tol = []
zero_normal, zeroone_normal, one_normal, ten_normal, twenty_normal = [], [], [], [], []
zero_normal_mean, zeroone_normal_mean, one_normal_mean, ten_normal_mean, twenty_normal_mean = [], [], [], [], []
zero_speed, zeroone_speed, one_speed, ten_speed, twenty_speed = [], [], [], [], []
zero_angle, zeroone_angle, one_angle, ten_angle, twenty_angle = [], [], [], [], []

id_num = [0, 0.1, 1, 10, 20]
tan = 0.09
length = 0

for a in range(len(id_num)):
    sdate = []
    file = r'D:\AI\Childfly\PS\result\{}'.format(id_num[a])
    name = []
    speed_pre = []
    time = []
    angle_pre = []
    length_tol_pre = []
    data = []
    angle_fre_pre = []
    for k in os.walk(file):
        for z in k[2]:
            name.append(z)
    for j in range(len(name)):
        ass = 0
        center = []
        file_dir = r'D:\AI\Childfly\PS\result\{}\{}'.format(id_num[a], name[j])
        with h5py.File(file_dir, "r") as f:
            locations = f["tracks"][:].T
        fps_num = locations.shape[0]
        fly_num = locations.shape[-1]
        date = pretreatment(a)
        for p in range(fly_num):
            speed = []
            angle = []
            angle_fre = 0
            length = 0
            for i in range(30, fps_num, 30):
                if not np.isnan(date[i, 0, 0, p]):
                    length += dist(date[i, 0, :, p], date[i - 30, 0, :, p]) * tan
                    speed.append(dist(date[i, 0, :, p], date[i - 30, 0, :, p]) * tan)
                    angle.append(GetAngle(date[i, 0, :, p], date[i - 15, 0, :, p], date[i - 15, 0, :, p],
                                          date[i - 30, 0, :, p]))
                    if 10 < GetAngle(date[i, 0, :, p], date[i - 15, 0, :, p], date[i - 15, 0, :, p],
                                     date[i - 30, 0, :, p]) <= 180:
                        angle_fre += 1
                else:
                    break
            speed_pre.append(speed)
            angle_pre.append(angle)
            length_tol_pre.append(length)
            time.append(len(speed))
            angle_fre_pre.append(angle_fre / len(angle))
    speed_tol.append(speed_pre)
    angle_tol.append(angle_pre)
    time_tol.append(time)
    length_tol.append(length_tol_pre)
    angle_fre_tol.append(angle_fre_pre)

Speed_Odd = draw_speed_odd()
Angle_Odd = draw_angle_odd()
Length = length_tol
Time = time_tol

