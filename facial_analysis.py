import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import dlib
from collections import OrderedDict
import math

class MTCNNFaceDetector(nn.Module):
    """Multi-task CNN for face detection and alignment"""
    def __init__(self):
        super(MTCNNFaceDetector, self).__init__()
        
        # P-Net (Proposal Network)
        self.pnet = PNet()
        
        # R-Net (Refine Network)
        self.rnet = RNet()
        
        # O-Net (Output Network)
        self.onet = ONet()
        
    def forward(self, image):
        # Multi-stage detection
        boxes_stage1 = self.pnet(image)
        boxes_stage2 = self.rnet(boxes_stage1)
        boxes_stage3, landmarks = self.onet(boxes_stage2)
        
        return boxes_stage3, landmarks

class PNet(nn.Module):
    """Proposal Network for face detection"""
    def __init__(self):
        super(PNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()
        
        # Classification
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        
        # Bounding box regression
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        
        cls = self.conv4_1(x)
        box = self.conv4_2(x)
        
        return cls, box

class RNet(nn.Module):
    """Refine Network for face detection"""
    def __init__(self):
        super(RNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1)
        self.prelu3 = nn.PReLU()
        
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.prelu4 = nn.PReLU()
        
        # Classification
        self.fc2_1 = nn.Linear(128, 2)
        
        # Bounding box regression
        self.fc2_2 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.prelu4(self.fc1(x))
        
        cls = self.fc2_1(x)
        box = self.fc2_2(x)
        
        return cls, box

class ONet(nn.Module):
    """Output Network for face detection and landmark localization"""
    def __init__(self):
        super(ONet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.prelu4 = nn.PReLU()
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.prelu5 = nn.PReLU()
        
        # Classification
        self.fc2_1 = nn.Linear(256, 2)
        
        # Bounding box regression
        self.fc2_2 = nn.Linear(256, 4)
        
        # Landmark localization
        self.fc2_3 = nn.Linear(256, 10)  # 5 landmarks * 2 coordinates
        
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))
        
        x = x.view(x.size(0), -1)
        x = self.prelu5(self.fc1(x))
        
        cls = self.fc2_1(x)
        box = self.fc2_2(x)
        landmarks = self.fc2_3(x)
        
        return cls, box, landmarks

class EmotionClassifier(nn.Module):
    """Emotion classification from facial features"""
    def __init__(self, num_emotions=7):
        super(EmotionClassifier, self).__init__()
        
        # ResNet18 backbone
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Emotion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        emotions = self.classifier(features)
        return emotions

class AgeGenderClassifier(nn.Module):
    """Age and gender classification from facial features"""
    def __init__(self):
        super(AgeGenderClassifier, self).__init__()
        
        # ResNet18 backbone
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Age regression
        self.age_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # Gender classification
        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        age = self.age_regressor(features)
        gender = self.gender_classifier(features)
        return age, gender

class GazeEstimator(nn.Module):
    """Gaze direction estimation"""
    def __init__(self):
        super(GazeEstimator, self).__init__()
        
        # ResNet18 backbone
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Gaze regression (pitch, yaw)
        self.gaze_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # pitch, yaw
        )
        
    def forward(self, x):
        features = self.backbone(x)
        gaze = self.gaze_regressor(features)
        return gaze

class FacialLandmarkDetector(nn.Module):
    """68-point facial landmark detection"""
    def __init__(self):
        super(FacialLandmarkDetector, self).__init__()
        
        # ResNet18 backbone
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Landmark regression
        self.landmark_regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 136)  # 68 landmarks * 2 coordinates
        )
        
    def forward(self, x):
        features = self.backbone(x)
        landmarks = self.landmark_regressor(features)
        return landmarks.view(-1, 68, 2)

class FaceAligner:
    """Face alignment using facial landmarks"""
    def __init__(self):
        # Template landmarks for alignment
        self.template_landmarks = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face using similarity transform"""
        # Select 5 key landmarks (eyes, nose, mouth corners)
        key_landmarks = landmarks[[36, 45, 30, 48, 54]]
        
        # Calculate similarity transform
        transform = cv2.estimateAffinePartial2D(
            key_landmarks, self.template_landmarks
        )[0]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(image, transform, (112, 112))
        
        return aligned_face

class FaceAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize models
        self.face_detector = MTCNNFaceDetector().to(device)
        self.emotion_classifier = EmotionClassifier().to(device)
        self.age_gender_classifier = AgeGenderClassifier().to(device)
        self.gaze_estimator = GazeEstimator().to(device)
        self.landmark_detector = FacialLandmarkDetector().to(device)
        
        # Face aligner
        self.face_aligner = FaceAligner()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Emotion labels
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        
    def analyze(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze all faces in the image"""
        # Detect faces
        faces = self.detect_faces(image)
        
        results = []
        for face in faces:
            # Extract face region
            face_image = self.extract_face_region(image, face['bbox'])
            
            # Detect landmarks
            landmarks = self.detect_landmarks(face_image)
            
            # Align face
            aligned_face = self.face_aligner.align_face(face_image, landmarks)
            
            # Analyze emotions
            emotions = self.analyze_emotions(aligned_face)
            
            # Analyze age and gender
            age, gender = self.analyze_age_gender(aligned_face)
            
            # Estimate gaze
            gaze = self.estimate_gaze(aligned_face)
            
            # Analyze micro-expressions
            micro_expressions = self.analyze_micro_expressions(aligned_face, landmarks)
            
            results.append({
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'landmarks': landmarks,
                'emotions': emotions,
                'age': age,
                'gender': gender,
                'gaze_direction': gaze,
                'micro_expressions': micro_expressions
            })
            
        return results
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image"""
        # Convert to tensor
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        input_tensor = input_tensor.to(self.device) / 255.0
        
        # Detect faces
        with torch.no_grad():
            boxes, landmarks = self.face_detector(input_tensor)
            
        faces = []
        if boxes is not None:
            for box in boxes:
                faces.append({
                    'bbox': box[:4].cpu().numpy(),
                    'confidence': box[4].cpu().numpy()
                })
                
        return faces
    
    def extract_face_region(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract face region from image"""
        x1, y1, x2, y2 = bbox.astype(int)
        face_region = image[y1:y2, x1:x2]
        return face_region
    
    def detect_landmarks(self, face_image: np.ndarray) -> np.ndarray:
        """Detect facial landmarks"""
        # Preprocess
        input_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # Detect landmarks
        with torch.no_grad():
            landmarks = self.landmark_detector(input_tensor)
            
        return landmarks.squeeze(0).cpu().numpy()
    
    def analyze_emotions(self, face_image: np.ndarray) -> Dict[str, float]:
        """Analyze facial emotions"""
        # Preprocess
        input_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # Classify emotions
        with torch.no_grad():
            emotion_logits = self.emotion_classifier(input_tensor)
            emotion_probs = F.softmax(emotion_logits, dim=1)
            
        # Convert to dictionary
        emotions = {}
        for i, emotion in enumerate(self.emotion_labels):
            emotions[emotion] = emotion_probs[0, i].item()
            
        return emotions
    
    def analyze_age_gender(self, face_image: np.ndarray) -> Tuple[float, str]:
        """Analyze age and gender"""
        # Preprocess
        input_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # Predict age and gender
        with torch.no_grad():
            age_pred, gender_logits = self.age_gender_classifier(input_tensor)
            gender_probs = F.softmax(gender_logits, dim=1)
            
        age = age_pred.item()
        gender = 'male' if gender_probs[0, 1] > 0.5 else 'female'
        
        return age, gender
    
    def estimate_gaze(self, face_image: np.ndarray) -> Tuple[float, float]:
        """Estimate gaze direction"""
        # Preprocess
        input_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # Estimate gaze
        with torch.no_grad():
            gaze_pred = self.gaze_estimator(input_tensor)
            
        pitch, yaw = gaze_pred.squeeze(0).cpu().numpy()
        return float(pitch), float(yaw)
    
    def analyze_micro_expressions(self, face_image: np.ndarray, 
                                 landmarks: np.ndarray) -> Dict[str, float]:
        """Analyze micro-expressions using facial landmarks"""
        # Calculate facial action units (AU) from landmarks
        action_units = self.calculate_action_units(landmarks)
        
        # Map action units to micro-expressions
        micro_expressions = {
            'eyebrow_raise': action_units.get('AU1', 0) + action_units.get('AU2', 0),
            'eye_widen': action_units.get('AU5', 0),
            'cheek_raise': action_units.get('AU6', 0),
            'lip_corner_pull': action_units.get('AU12', 0),
            'dimpler': action_units.get('AU14', 0),
            'lip_corner_depress': action_units.get('AU15', 0),
            'chin_raise': action_units.get('AU17', 0),
            'lip_pucker': action_units.get('AU18', 0)
        }
        
        return micro_expressions
    
    def calculate_action_units(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate facial action units from landmarks"""
        # This is a simplified implementation
        # In practice, use more sophisticated AU detection
        
        action_units = {}
        
        # Eye region (landmarks 36-47)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Calculate eye openness
        left_eye_height = np.mean(left_eye[1::2, 1]) - np.mean(left_eye[::2, 1])
        right_eye_height = np.mean(right_eye[1::2, 1]) - np.mean(right_eye[::2, 1])
        action_units['AU5'] = (left_eye_height + right_eye_height) / 2
        
        # Eyebrow region (landmarks 17-26)
        eyebrows = landmarks[17:27]
        eyebrow_height = np.mean(eyebrows[:, 1])
        action_units['AU1'] = eyebrow_height
        action_units['AU2'] = eyebrow_height
        
        # Mouth region (landmarks 48-67)
        mouth = landmarks[48:68]
        mouth_width = np.linalg.norm(mouth[6] - mouth[0])
        mouth_height = np.linalg.norm(mouth[9] - mouth[3])
        
        action_units['AU12'] = mouth_width
        action_units['AU15'] = -mouth_height  # Negative for depression
        action_units['AU18'] = mouth_height / mouth_width
        
        # Cheek region
        left_cheek = landmarks[1]
        right_cheek = landmarks[15]
        cheek_height = (left_cheek[1] + right_cheek[1]) / 2
        action_units['AU6'] = cheek_height
        
        return action_units

class FacialAnalysisTrainer:
    """Training pipeline for facial analysis models"""
    def __init__(self, device='cuda'):
        self.device = device
        
    def train_emotion_classifier(self, model, dataloader, optimizer, criterion):
        """Train emotion classifier"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train_age_gender_classifier(self, model, dataloader, optimizer, 
                                   age_criterion, gender_criterion):
        """Train age and gender classifier"""
        model.train()
        total_loss = 0
        age_error = 0
        gender_correct = 0
        total = 0
        
        for batch_idx, (data, age_target, gender_target) in enumerate(dataloader):
            data = data.to(self.device)
            age_target = age_target.to(self.device)
            gender_target = gender_target.to(self.device)
            
            optimizer.zero_grad()
            age_output, gender_output = model(data)
            
            age_loss = age_criterion(age_output.squeeze(), age_target)
            gender_loss = gender_criterion(gender_output, gender_target)
            
            total_loss = age_loss + gender_loss
            total_loss.backward()
            optimizer.step()
            
            # Calculate metrics
            age_error += torch.abs(age_output.squeeze() - age_target).sum().item()
            _, gender_predicted = gender_output.max(1)
            gender_correct += gender_predicted.eq(gender_target).sum().item()
            total += age_target.size(0)
            
        mae = age_error / total
        gender_accuracy = 100. * gender_correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, mae, gender_accuracy

# Training configuration
FACIAL_ANALYSIS_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'weight_decay': 1e-5,
    'emotions': 7,
    'age_range': [0, 100],
    'genders': 2,
    'augmentation': {
        'rotation': 15,
        'brightness': 0.2,
        'contrast': 0.2,
        'horizontal_flip': 0.5
    }
}

# Evaluation metrics
def evaluate_facial_analysis(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Evaluate facial analysis results"""
    emotion_correct = 0
    age_error = 0
    gender_correct = 0
    gaze_error = 0
    total_faces = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        # Emotion accuracy
        pred_emotion = max(pred['emotions'], key=pred['emotions'].get)
        if pred_emotion == gt['emotion']:
            emotion_correct += 1
            
        # Age MAE
        age_error += abs(pred['age'] - gt['age'])
        
        # Gender accuracy
        if pred['gender'] == gt['gender']:
            gender_correct += 1
            
        # Gaze error
        gaze_error += np.sqrt(
            (pred['gaze_direction'][0] - gt['gaze_direction'][0])**2 +
            (pred['gaze_direction'][1] - gt['gaze_direction'][1])**2
        )
    
    return {
        'emotion_accuracy': emotion_correct / total_faces,
        'age_mae': age_error / total_faces,
        'gender_accuracy': gender_correct / total_faces,
        'gaze_mae': gaze_error / total_faces
    }