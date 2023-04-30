from django.shortcuts import render
from django.contrib import messages
from .forms import Ques_ans_form, Image_form, Video_form
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import cv2
import torchvision
from torchvision import transforms as torchtrans  
import matplotlib.pyplot as plt
import time
# Create your views here.
import tensorflow as tf
from io import BytesIO
from keras.models import load_model
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from torch.utils.data import Dataset, DataLoader
plt.switch_backend('Agg')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

model2 = load_model("app/model_casia_run1.h5")
image_size = (128, 128)
cpu_device = torch.device("cpu")

model3 = torch.load("app/gun_detection_model.pt", map_location=torch.device('cpu'))



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class gun(Dataset):
    def __init__(self,imgs_path,labels_path):
        self.imgs = imgs_path

    def __getitem__(self,idx):

        img = convert_from_image_to_cv2(self.imgs[0])
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = img_rgb/255
        img_res = torch.as_tensor(img_res).to(device)
        img_res = img_res.permute(2, 0, 1)

        return img_res

    def __len__(self):
        return len(self.imgs)


def convert_to_ela_image(img, quality):
    temp_filename = 'temp_file_name.jpg'
    
    image = img.convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale).resize(image_size)
    return ela_image

def prepare_image(img):
    return np.array(convert_to_ela_image(img, 90).resize(image_size)).flatten() / 255.0


def question_answer(question, text):
    
    #tokenize question and text in ids as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)

    #number of tokens in segment A - question
    num_seg_a = sep_idx+1

    #number of tokens in segment B - text
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    
    # assert len(segment_ids) == len(input_ids)
    
    # #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    # #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    answer=""
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    # if answer.startswith("[CLS]"):
    #     answer = "Unable to find the answer to your question."
    
#     print("Text:\n{}".format(text.capitalize()))
#     print("\nQuestion:\n{}".format(question.capitalize()))
    return answer

def ques_ans(request):
    if request.method == "POST":
        text = request.POST.get('text')
        question = request.POST.get('question')
        print(question, text)
        #send text question to model
        messages.info(request, question_answer(question, text))
    
    form = Ques_ans_form()
    return render(request, 'app/ques_ans.html', {'form':form})

def splice_detect(request):
    if request.method == "POST":
        #send image to model
        f = Image_form(request.POST, request.FILES)
        if f.is_valid():
            image = Image.open(BytesIO(request.FILES['image'].read()))
            ela_image = prepare_image(image)
            ela_image = ela_image.reshape(-1, 128, 128, 3)
            val = model2.predict(ela_image)[0][0]
            messages.info(request, f"probability of manipulation is {val:.2f}")

    form = Image_form()
    return render(request, 'app/splice_detect.html', {'form':form})

def convert_from_image_to_cv2(img) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def apply_nms(orig_prediction, iou_thresh=0.7):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    
    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')


def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    img = convert_from_image_to_cv2(img)
    for box in (target['boxes']):
        box = box.detach().numpy()
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        # print((box[0], box[1]), (box[2], box[3]))
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), thickness=2, color=(255, 0, 0))
        # rect = patches.Rectangle((x, y),
                                #  width, height,
                                #  linewidth = 2,
                                #  edgecolor = 'r',
                                #  facecolor = 'none')

        # Draw the bounding box on top of the image
        # a.add_patch(rect)
    # plt.savefig('app/static/app/final_output.png')
    return img

def generate_video(images):
    height, width, layers = images[0].shape  
  
    video = cv2.VideoWriter('app/static/app/output_video.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (width, height))  
  
    # Appending the images to the video one by one
    for image in images: 
        print("inside")
        video.write(image) 
      
    # Deallocating memories taken for window creation
    print("done")
    cv2.destroyAllWindows() 
    video.release()


def handle_uploaded_file(f):
    with open("video.mp4", 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    cap = cv2.VideoCapture("/Users/bhavyashah/Major-project/video.mp4")
    success, img = cap.read()
    fno = 0
    arr = []
    while success:
        # read next frame
        # image = Image.open(BytesIO(request.FILES['image'].read()))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fno+=1
        image = Image.fromarray(np.uint8(img))
        test_data = gun([image], [0])
        img = test_data[0]
        input = []
        input.append(img)
        outputs = model3(input)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        nms_prediction = apply_nms(outputs[0], iou_thresh=0.7)
        print('NMS APPLIED MODEL OUTPUT', fno)
        arr.append(plot_img_bbox(torch_to_pil(img), nms_prediction))
        success, img = cap.read()
    print("here")
    generate_video(arr)

def weapons_detect(request):
    show = False
    if request.method == "POST":
        file = request.FILES['video']
        handle_uploaded_file(file)
        show = True
    print(show)
    form = Video_form()
    return render(request, 'app/weapons_detect.html', {'form':form, 'show': show})
