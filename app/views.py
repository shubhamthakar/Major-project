from django.shortcuts import render
from django.contrib import messages
from .forms import Ques_ans_form, Image_form
# Create your views here.

def ques_ans(request):
    if request.method == "POST":
        text = request.POST.get('text')
        question = request.POST.get('question')
        print(question, text)
        #send text question to model
        messages.info(request, "Show output here")
    
    form = Ques_ans_form()
    return render(request, 'app/ques_ans.html', {'form':form})

def splice_detect(request):
    if request.method == "POST":
        #send image to model
        image = request.POST.get('image')
        print(image)
        messages.info(request, "Show output here")

    form = Image_form()
    return render(request, 'app/splice_detect.html', {'form':form})

def weapons_detect(request):
    if request.method == "POST":
        #send image to model
        image = request.POST.get('image')
        print(image)
        messages.info(request, "Show output here")

    form = Image_form()
    return render(request, 'app/weapons_detect.html', {'form':form})
