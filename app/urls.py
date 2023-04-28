from django.urls import path
from . import views

urlpatterns = [
    path('question-answering-tool/',views.ques_ans, name = "ques_ans"),
    path('spliced-image-detection/',views.splice_detect, name= "splice_detect"),
    path('weapons-detection/',views.weapons_detect,name='weapons_detect')
]
