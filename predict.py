import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

device = torch.device("cpu")

def ColorPredict(_model, image_path, classes):

    with st.spinner('Predicting color, please wait...'):

        # switch the model to evaluation mode to make dropout and batch norm work in eval mode
        _model.eval()

        # transforms for the input image
        loader = transforms.Compose([transforms.Resize((400, 400)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = Image.open(image_path)
        image = loader(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        output = _model(image)
        conf, predicted = torch.max(output.data, 1)
        # get the class name of the prediction
        return classes[predicted.item()], round(conf.item(),2)


def MakeModelPredict(_model, image_path, classes):

    with st.spinner('Predicting make and model, please wait...'):

        # switch the model to evaluation mode to make dropout and batch norm work in eval mode
        _model.eval()

        # transforms for the input image
        loader = transforms.Compose([transforms.Resize((400, 400)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = Image.open(image_path)
        image = loader(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        output = _model(image)
        conf, predicted = torch.max(output.data, 1)
        # get the class name of the prediction
        return classes[predicted.item()], round(conf.item(),2)


def TypePredict(_model, img_path, classes):

    with st.spinner('Predicting car type, please wait...'):

        # switch the model to evaluation mode to make dropout and batch norm work in eval mode
        _model.eval()

        # transforms for the input image
        loader = transforms.Compose([transforms.Resize((400, 400)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = Image.open(img_path)
        image = loader(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        output = _model(image)
        conf, predicted = torch.max(output.data, 1)
        # get the class name of the prediction
        return classes[predicted.item()], round(conf.item(),2)
