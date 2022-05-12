#    line = [(80,290),(510,170)] #Solsiden
def adaptiveLine(data):
    numberOfPointsStart = 0
    numberOfPointsEnd = 0
    startX =0
    startY = 0
    endX = 0
    endY = 0

    for i in range(1,len(data)):
        if len(data[i]) >0:
            if (data[i][0][0]) < 250:                
                startX += (data[i][0][0])
                startY += (data[i][0][1])
                numberOfPointsStart +=1

            elif (data[i][0][0]) > 400:                
                endX += (data[i][0][0])
                endY += (data[i][0][1])
                numberOfPointsEnd +=1

            elif (data[i][-1][0]) > 400:
                endX += (data[i][-1][0])
                endY += (data[i][-1][1])
                numberOfPointsEnd +=1

            elif (data[i][-1][0]) < 250:                
                startX += (data[i][-1][0])
                startY += (data[i][-1][1])
                numberOfPointsStart +=1

    startPoint = (80,(int(startY/numberOfPointsStart)+200))
    endPoint = (510,(int(endY/numberOfPointsEnd)+50))
    line = [startPoint,endPoint]
    return line


if __name__ == '__main__':
    pass

    #            data = {}
    #        for i in range(1, highestTrackId+1):
    #            row = []
    #            for j in range(0, len(history[i])):
    #                if history[i][j] is None:
    #                    continue
    #                row.append(history[i][j])
    #            data[i] =row