import turtle as t 
#turtle이 기니까 t라고하겠다.

#마커 정의 영역
class MagicBrush:
    t.color('red')
    def draw_squre(self):# 명시적, 드래그 후 탭으로 전체에 넣을 수 있다.
        for i in range(4):#명시적이지 않다.
            t.forward(100)
            t.right(90)
    def draw_triangle(self):
        for i in range(3):
            t.forward(100)
            t.right(120)
    def go(self):
        t.forward(200)
    def turn(self):
        t.right(90)
    def draw_hexagon(self):
        for i in range(6):
            t.forward(100)
            t.right(60)
#m1 = MagicBrush()#만들고
#m2 = MagicBrush()
brad = t.Turtle()
brad.shape('turtle')
brad.speed(2)
brad.forward(100)

t.mainloop()