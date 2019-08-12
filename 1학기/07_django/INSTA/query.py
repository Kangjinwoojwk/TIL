from mton.models import Student, Lecture, Enrolment

create_student = Student.objects.create
s1 = create_student(name='aaa')
s2 = create_student(name='bbb')
s3 = create_student(name='ccc')

l1 = Lecture.objects.create(title='컴개')
l2 = Lecture.objects.create(title='자료구조')
l3 = Lecture.objects.create(title='알고리즘')

Enrolment.objects.create(lecture=l1, student=s1)  # django가 알아서 인식
Enrolment.objects.create(lecture=l1, student=s2)
Enrolment.objects.create(lecture=l1, student=s3)



이종화 = Student.objects.get(id=1)
이종화.enrolment_set.all()  # id=1, lecture_id=1