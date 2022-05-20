import cv2
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy

def reader(dir):
    df = pd.read_csv(f'{dir}/data.csv', na_filter=False)
    rows, cols = df.shape
    data = {}
    for i in range(0,rows):
        row = []
        for j in range(1,cols):
            if len(df.iat[i,j]) > 1:
                row.append(ast.literal_eval(df.iat[i,j]))   
        data[i+1] = row
    return data

def splitTracjetoryInEmptyGraph(data,dir):
    walking = {}
    falling = {}
    wIndex = 1
    fIndex = 1
    for i in range(1,len(data)+1):
        if len(data[i]) >0:
            x,y = (data[i][-1])
            if y > 160 and data[i][0][1] < y and x > 150 or data[i][0][1] > 200 and y > 200 and x > 150 and data[i][0][1] < y:
                falling[fIndex] = data[i]
                fIndex +=1
            else:
                walking[wIndex] = data[i]
                wIndex +=1
    tracjetoryInEmptyGraph(falling,dir,"Trajectories from fall")
    tracjetoryInEmptyGraph(walking,dir,"Non falling trajectories")

def tracjetoryInEmptyGraph(data,dir,name):
  # initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    img =  255 * np.ones(shape=[512, 640, 3], dtype=np.uint8)
    for i in range(1,len(data)+1):
        color = colors[int(i) % len(colors)]
        color = [i * 255 for i in color]
        for j in range(1, len(data[i])):
            if data[i][j - 1] is None or data[i][j] is None:
                continue
            cv2.line(img,(data[i][j-1]), (data[i][j]),(color),1)

    plt.gca().invert_yaxis()
    plt.title(name)
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel value")
    plt.imshow(img)
    plt.savefig(f'{dir}/{name}.png')
    plt.show()


def distancePlot(data,dir):
    distance = []
    val = {}
    for i in range(1,len(data)+1):
        for j in range(1, len(data[i])):
            if data[i][j - 1] is None or data[i][j] is None:
                continue
            point1x,point1y = data[i][j-1]
            point2x,point2y = data[i][j]
            dis = np.sqrt((point2x - point1x)**2 + (point2y - point1y)**2)
            distance.append(round(dis,2))
    
    val["x"] = distance
    sns.set_theme(style="whitegrid")
    sns.boxenplot(x=val["x"]).set(xlabel='Distance')
    plt.grid(False)
    plt.title("Distance plot")
    plt.savefig(f'{dir}/distancePlot.png')
    plt.show()

def trajectoryPlot(data,dir):
    x =[]
    y = []
    for i in range(1,len(data)+1):
        for j in range(0, len(data[i])):
            x.append(data[i][j][0])
            y.append(data[i][j][1])
    val = [x,y]
    sns.set_theme(style="whitegrid")
    
    sns.boxenplot(data=val).set(xlabel='<- X                Y ->')
    plt.grid(False)
    plt.title("Number of times a coordinate is present")
    plt.savefig(f'{dir}/trajectoryPlot.png')
    plt.show()
    

def trajectoryMap(data,dir):
    # initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    img = cv2.imread(f'{dir}/pic.png')
    for i in range(1,len(data)+1):
        color = colors[int(i) % len(colors)]
        color = [i * 255 for i in color]
        for j in range(1, len(data[i])):
            if data[i][j - 1] is None or data[i][j] is None:
                continue
            cv2.line(img,(data[i][j-1]), (data[i][j]),(color),1)
    cv2.imshow("Trajectory map", img)
    cv2.waitKey(0)
    cv2.imwrite(f'{dir}/trajectoryMap.png',img)

def heatMap(data,dir):
    x =[]
    y = []
    for i in range(1,len(data)+1):
        for j in range(0, len(data[i])):
            x.append(data[i][j][0])
            y.append(data[i][j][1])
    
    cmap = copy(plt.get_cmap('hot'))
    cmap.set_bad(cmap(0))
    plt.grid(False)
    h = plt.hist2d(x,y, bins=[np.arange(0,640,3),np.arange(0,512,3)],cmap=cmap)
    plt.title("Heatmap")
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.colorbar(h[3])
    plt.savefig(f'{dir}/heatMap.png')
    plt.show()

def distanceMap(data,dir):
    x =[]
    y = []
    for i in range(1,len(data)+1):
        for j in range(0, len(data[i])):
            x.append(data[i][j][0])
            y.append(data[i][j][1])

    distance = []
    for i in range(1,len(data)+1):
        for j in range(0, len(data[i])):
            if data[i][j - 1] is None or data[i][j] is None:
                continue
            point1x,point1y = data[i][j-1]
            point2x,point2y = data[i][j]
            dis = np.sqrt((point2x - point1x)**2 + (point2y - point1y)**2)
            distance.append(round(dis,2))

    cmap = copy(plt.get_cmap('hot'))
    cmap.set_bad(cmap(0))
    plt.title("Distance between two points")
    plt.scatter(x, y, c=distance,cmap=cmap)
    plt.grid(False)
    plt.colorbar()
    plt.xlim([0, 640])
    plt.ylim([0, 512])
    plt.gca().invert_yaxis()
    plt.savefig(f'{dir}/distanceMap.png')
    plt.show()


def meanTrajcetory(data,dir,name):
  # initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    img =  255 * np.ones(shape=[512, 640, 3], dtype=np.uint8)
    for i in range(1,len(data)+1):
        color = colors[int(i) % len(colors)]
        color = [i * 255 for i in color]
        for j in range(1, len(data[i])):
            if data[i][j - 1] is None or data[i][j] is None:
                continue
            cv2.line(img,(data[i][j-1]), (data[i][j]),(color),1)

    numberOfPointsStart = 0
    numberOfPointsEnd = 0

    startX =0
    startY = 0
    endX = 0
    endY = 0
    for i in range(1,len(data)+1):
        if len(data[i]) >0:
            if (data[i][0][0]) < 140:                
                startX += (data[i][0][0])
                startY += (data[i][0][1])
                numberOfPointsStart +=1

            elif (data[i][0][0]) > 300 and data[i][0][1] < 160:                
                endX += (data[i][0][0])
                endY += (data[i][0][1])
                numberOfPointsEnd +=1

            elif (data[i][-1][0]) > 300 and data[i][-1][1] < 160:
                endX += (data[i][-1][0])
                endY += (data[i][-1][1])
                numberOfPointsEnd +=1

            elif (data[i][-1][0]) < 140:                
                startX += (data[i][-1][0])
                startY += (data[i][-1][1])
                numberOfPointsStart +=1

    startPoint = (0,int(startY/numberOfPointsStart))
    endPoint = (640,int(endY/numberOfPointsEnd))
    cv2.line(img,startPoint, endPoint,(0,255,0),2)
    plt.gca().invert_yaxis()
    plt.title(name)
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel value")
    plt.imshow(img)
    plt.savefig(f'{dir}/{name}.png')
    plt.show()
    
def oneDPlot(data,dir):
    for i in [9,69,72,78]:
    #for i in range(1,len(data)+1):
        x =[]
        y = []
        if len(data[i])>10:
            #i= 169
            print(i)
            for j in range(0, len(data[i])):
                x.append(data[i][j][0])
                y.append(data[i][j][1])

            plt.plot(x,y)
            plt.gca().invert_yaxis()
            plt.title("1d plot non falling")
            plt.xlabel("x position")
            plt.ylabel("y position")
            plt.savefig(f'{dir}/1dPlotWalking2.png')
            plt.show()

            time = []
            for j in range(1, len(data[i])+1):
                time.append(j)

            plt.plot(time,x)
            plt.plot(time,y)
            plt.legend(["X","Y"])
            plt.gca().invert_yaxis()
            plt.title("Position versus frames (non falling persons)")
            plt.xlabel("time (frame number)")
            plt.ylabel("position")
            plt.savefig(f'{dir}/xySplitWalking2.png')
            plt.show()

            img =  255 * np.ones(shape=[512, 640, 3], dtype=np.uint8)
            for j in range(1, len(data[i])):
                if data[i][j - 1] is None or data[i][j] is None:
                    continue
                cv2.line(img,(data[i][j-1]), (data[i][j]),(0,0,255),1)
            plt.gca().invert_yaxis()
            plt.title("Non falling trajcetory")
            plt.xlabel("Pixel value")
            plt.ylabel("Pixel value")
            plt.imshow(img)
            plt.savefig(f'{dir}/1dTrajcetoryWalking2.png')
            plt.show()

def comparePlot(data,dir):
    walk = [9,69,72,78]
    for i in walk:
        x1 =[]
        y1 = []
        for j in range(0, len(data[i])):
            x1.append(data[i][j][0])
            y1.append(data[i][j][1])
    
        time = []
        for j in range(1, len(data[i])+1):
        
            time.append(j)
        plt.plot(time,x1,'y--')
        plt.plot(time,y1,'r')
    
    fall =[16,39,52,65]
    for i in fall:
        x2 =[]
        y2 = []
        for j in range(0, len(data[i])):
            x2.append(data[i][j][0])
            y2.append(data[i][j][1])

        time1 = []
        for j in range(1, len(data[i])+1):
            time1.append(j)
        plt.plot(time1,x2,"g--")
        plt.plot(time1,y2,"b")

    lines = plt.gca().get_lines()
    legend2 = plt.legend([lines[i] for i in [0,1,8,9]],['Walk X','Walk Y','Fall X','Fall Y'], loc=4)
    plt.gca().invert_yaxis()
    plt.title("Position versus frames (falling and non falling)")
    plt.xlabel("time (frame number)")
    plt.ylabel("position")
    plt.savefig(f'{dir}/compare.png')
    plt.show()

if __name__ == '__main__':
    dir = 'C:/Users/peda_/Documents/Masteroppgave/deepSort/viz/test'
    data = reader(dir)
    #trajectoryMap(data,dir)
    #trajectoryPlot(data,dir)
    #heatMap(data,dir)
    #distanceMap(data,dir)
    #distancePlot(data,dir)
    #tracjetoryInEmptyGraph(data,dir,"All trajectories")
    #splitTracjetoryInEmptyGraph(data,dir)
    #meanTrajcetory(data,dir,"Mean trajcetory")
    #oneDPlot(data,dir)
    comparePlot(data,dir)