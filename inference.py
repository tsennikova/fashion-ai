import torch
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn

MODEL_PATH = "models/stage-1_sz-224_torch.pkl"
DATA_PATH = "deepfashion-dataset/"
CLASSES_PATH = "/clothes_categories/classes.txt"

class ClassificationModel():
    
    def __init__(self):
        return
        
    def load(self, model_path, labels_path,  eval=False):
        self.model = torch.load(model_path)
        self.model = nn.Sequential(self.model)
        
        self.labels = open(labels_path, 'r').read().splitlines()
        
        if eval:
            print(model.eval())
        return
    
    def predict(self, image_path):
        
        device = torch.device("cpu")
        img = Image.open(image_path)
        
        test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])
        
        image_tensor = test_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        inp = inp.to(device)
        output = self.model(inp)
        index = output.data.cpu().numpy().argmax()
        return self.labels[index]

learner = ClassificationModel()
learner.load(MODEL_PATH, CLASSES_PATH)
learner.predict(DATA_PATH+"img-phone-jpg\IMG_2966.jpg")