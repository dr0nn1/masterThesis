#made from this: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D) 

if __name__ == '__main__':
    point1 = (0,0)
    point2 = (2,2)
    point3 = (0,3)
    point4 = (3,0)
    print(intersect(point1,point2,point3,point4))