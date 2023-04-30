from django import forms

class Ques_ans_form(forms.Form):
    text = forms.CharField(label = 'text')
    question = forms.CharField(label = 'question')

class Image_form(forms.Form):
    image = forms.ImageField()


class Video_form(forms.Form):
    video = forms.FileField()