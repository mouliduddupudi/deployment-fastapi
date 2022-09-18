from fastapi import FastAPI , File , UploadFile 
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn 
import numpy as np 
from io import BytesIO 
from PIL import Image 
import torch
from CNN import CNN , idx_to_classes
from torchvision import transforms  

app = FastAPI() 

origins = [
    "http://localhost",
    "http://localhost:3000",
] 

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True, 
    allow_methods = ["*"], 
    allow_headers = ["*"],
) 

CLASS_NAMES =  [ 
    "Arborio", 'Basmati','Ipsala', 'Jasmine','Karacadag'
]

def prediction(image_obj , idx_to_classes): 
    INPUT_DIM = 224 

    preprocess = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])

    pretrained_model = CNN(5) 
    pretrained_model.load_state_dict( 
        torch.load('model.pth' , map_location=torch.device('cpu'))  
    ) 

    im = image_obj 
    im_preprocessed = preprocess(im) 
    batch_img_tensor = torch.unsqueeze(im_preprocessed , 0) 
    output = pretrained_model(batch_img_tensor)
    output = output.detach().numpy()  
    index = np.argmax(output) 
    predicted_class = idx_to_classes[index] 
    confidence = np.max(output[0])  
    return predicted_class , confidence*100


@app.get("/rice_grains")
async def get_grains():
    return {"rice_grains": CLASS_NAMES}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
): 
    image = Image.open(file.file)
    print("type" , type(image))
    result = prediction(image , idx_to_classes)
    return {
        "img_batch" : str(type(image)) ,
        "prediction": result[0], 
        "confidence": result[1]
    }
