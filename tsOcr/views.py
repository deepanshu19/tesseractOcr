from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import viewsets
from rest_framework.response import Response
from .models import Ocr
from . import serializers
from rest_framework.decorators import api_view

# import pytesseract to convert text in image to string
import pytesseract
from pytesseract import Output
import pandas as pd
import cv2

# import summarize to summarize the ocred text
from gensim.summarization.summarizer import summarize

from .forms import ImageUpload
import os

# import Image from PIL to read image
from PIL import Image

from django.conf import settings


# Create your views here.
def output_file(filename, data):
    file = open(filename, "w+")
    file.write(data)
    file.close()


def index(request):

    text1 = ""
    summarized_text = ""
    message = ""
    if request.method == 'POST':
        form = ImageUpload(request.POST, request.FILES)
        if form.is_valid():
            try:
                form.save()
                image = request.FILES['image']
                image = image.name
                path = settings.MEDIA_ROOT
                pathz = path + "/images/" + image
                img = cv2.imread(pathz)
                img = cv2.resize(img, (int(img.shape[1] + (img.shape[1] * .1)),
                                       int(img.shape[0] + (img.shape[0] * .25))),
                                 interpolation=cv2.INTER_AREA)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.$%./@& *"'
                d = pytesseract.image_to_data(
                    img_rgb, config=custom_config, output_type=Output.DICT)
                df = pd.DataFrame(d)

                # clean up blanks
                df1 = df[(df.conf != '-1') &
                         (df.text != ' ') & (df.text != '')]
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)

                # sort blocks vertically
                sorted_blocks = df1.groupby(
                    'block_num').first().sort_values('top').index.tolist()
                for block in sorted_blocks:
                    curr = df1[df1['block_num'] == block]
                    sel = curr[curr.text.str.len() > 3]
                    # sel = curr
                    char_w = (sel.width / sel.text.str.len()).mean()
                    prev_par, prev_line, prev_left = 0, 0, 0
                    text = ''
                    for ix, ln in curr.iterrows():
                        # add new line when necessary
                        if prev_par != ln['par_num']:
                            text += '\n'
                            prev_par = ln['par_num']
                            prev_line = ln['line_num']
                            prev_left = 0
                        elif prev_line != ln['line_num']:
                            text += '\n'
                            prev_line = ln['line_num']
                            prev_left = 0

                        added = 0  # num of spaces that should be added
                        if ln['left'] / char_w > prev_left + 1:
                            added = int((ln['left']) / char_w) - prev_left
                            text += ' ' * added
                        text += ln['text'] + ' '
                        prev_left += len(ln['text']) + added + 1
                    text += '\n'
                    print(text)

                    output_file("txtOutput.txt", text)
                    text1 = text

                # text = pytesseract.image_to_string(Image.open(pathz))
                # text = text.encode("ascii", "ignore")
                # text = text.decode()

                # Summary (0.1% of the original content).
                summarized_text = summarize(text, ratio=0.1)
                os.remove(pathz)
            except:
                message = "check your filename and ensure it doesn't have any space or check if it has any text"

    context = {
        'text': text1,
        'summarized_text': summarized_text,
        'message': message
    }
    return render(request, 'formpage.html', context)


@api_view(['GET'])
def getData(request):
    ocrs = Ocr.objects.all()
    oSerializer = serializers.ocrSerializer(ocrs, many=True)
    return Response(oSerializer.data)


class OcrToText(viewsets.ModelViewSet):
    queryset = Ocr.objects.all()
    serializer_class = serializers.ocrSerializer


@api_view(['GET'])
def getImage(request, name1):
    text1 = ''
    ocrs = Ocr.objects.get(name=name1)
    oSerializer = serializers.ocrImageSerializer(ocrs, many=False)
    print(oSerializer.data['image'])
    pathz = oSerializer.data['image']
    pathz = pathz[1:]
    img = cv2.imread(pathz)
    img = cv2.resize(img, (int(img.shape[1] + (img.shape[1] * .1)),
                           int(img.shape[0] + (img.shape[0] * .25))),
                     interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.$%./@& *"'
    d = pytesseract.image_to_data(
        img_rgb, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)

    # clean up blanks
    df1 = df[(df.conf != '-1') &
             (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # sort blocks vertically
    sorted_blocks = df1.groupby(
        'block_num').first().sort_values('top').index.tolist()
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        # sel = curr
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''
        for ix, ln in curr.iterrows():
            # add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0  # num of spaces that should be added
            if ln['left'] / char_w > prev_left + 1:
                added = int((ln['left']) / char_w) - prev_left
                text += ' ' * added
            text += ln['text'] + ' '
            prev_left += len(ln['text']) + added + 1
        text += '\n'
        # print(text)

        # output_file("txtOutput.txt", text)
        text1 = text
    return Response({'text': text1})
