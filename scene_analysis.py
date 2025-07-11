import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
import colorsys

class SceneClassifier(nn.Module):
    def __init__(self, num_classes=365, pretrained=True):
        super(SceneClassifier, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = resnet50(pretrained=pretrained)
        
        # Replace final layer for scene classification
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class LightingClassifier(nn.Module):
    def __init__(self, input_features=512, num_lighting_conditions=8):
        super(LightingClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_lighting_conditions)
        )
        
    def forward(self, x):
        return self.classifier(x)

class IndoorOutdoorClassifier(nn.Module):
    def __init__(self, input_features=512):
        super(IndoorOutdoorClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Indoor/Outdoor
        )
        
    def forward(self, x):
        return self.classifier(x)

class ColorAnalyzer:
    def __init__(self):
        self.color_names = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
            'yellow': [255, 255, 0],
            'cyan': [0, 255, 255],
            'magenta': [255, 0, 255],
            'orange': [255, 165, 0],
            'purple': [128, 0, 128],
            'pink': [255, 192, 203],
            'brown': [165, 42, 42],
            'gray': [128, 128, 128],
            'black': [0, 0, 0],
            'white': [255, 255, 255]
        }
        
    def extract_dominant_colors(self, image: np.ndarray, num_colors=5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image using K-means clustering"""
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # Calculate color frequencies
        labels = kmeans.labels_
        color_frequencies = []
        for i in range(num_colors):
            frequency = np.sum(labels == i) / len(labels)
            color_frequencies.append(frequency)
            
        # Sort by frequency
        sorted_indices = np.argsort(color_frequencies)[::-1]
        dominant_colors = dominant_colors[sorted_indices]
        
        return [(int(color[0]), int(color[1]), int(color[2])) for color in dominant_colors]
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get the closest color name for an RGB value"""
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for color_name, color_rgb in self.color_names.items():
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, color_rgb)))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
                
        return closest_color
    
    def analyze_color_harmony(self, colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Analyze color harmony and temperature"""
        if not colors:
            return {}
            
        # Convert to HSV for analysis
        hsv_colors = []
        for rgb in colors:
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hsv_colors.append(hsv)
            
        # Calculate color temperature
        warm_colors = 0
        cool_colors = 0
        
        for hsv in hsv_colors:
            hue = hsv[0] * 360
            if (hue >= 0 and hue <= 60) or (hue >= 300 and hue <= 360):
                warm_colors += 1
            elif hue >= 120 and hue <= 240:
                cool_colors += 1
                
        color_temperature = 'warm' if warm_colors > cool_colors else 'cool'
        if warm_colors == cool_colors:
            color_temperature = 'neutral'
            
        # Calculate color diversity
        hue_values = [hsv[0] for hsv in hsv_colors]
        hue_std = np.std(hue_values)
        color_diversity = 'high' if hue_std > 0.3 else 'low'
        
        return {
            'temperature': color_temperature,
            'diversity': color_diversity,
            'warm_ratio': warm_colors / len(colors),
            'cool_ratio': cool_colors / len(colors)
        }

class LightingAnalyzer:
    def __init__(self):
        self.lighting_conditions = [
            'natural_daylight', 'artificial_indoor', 'golden_hour',
            'blue_hour', 'overcast', 'harsh_shadows', 'soft_lighting', 'backlit'
        ]
        
    def analyze_lighting(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting conditions in the image"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness statistics
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Calculate contrast
        contrast = brightness_std / brightness_mean if brightness_mean > 0 else 0
        
        # Analyze histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()
        
        # Calculate histogram statistics
        hist_peak = np.argmax(hist_normalized)
        hist_spread = np.std(range(256), weights=hist_normalized.flatten())
        
        # Shadow and highlight analysis
        shadows = np.sum(gray < 85) / gray.size
        highlights = np.sum(gray > 170) / gray.size
        midtones = 1 - shadows - highlights
        
        # Color temperature analysis
        color_temp = self.estimate_color_temperature(image)
        
        return {
            'brightness_mean': float(brightness_mean),
            'brightness_std': float(brightness_std),
            'contrast': float(contrast),
            'shadows_ratio': float(shadows),
            'highlights_ratio': float(highlights),
            'midtones_ratio': float(midtones),
            'color_temperature': color_temp,
            'histogram_peak': int(hist_peak),
            'histogram_spread': float(hist_spread)
        }
    
    def estimate_color_temperature(self, image: np.ndarray) -> str:
        """Estimate color temperature of the image"""
        # Calculate average RGB values
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])
        
        # Calculate blue-red ratio
        br_ratio = avg_b / (avg_r + 1e-6)
        
        if br_ratio > 1.1:
            return 'cool'
        elif br_ratio < 0.9:
            return 'warm'
        else:
            return 'neutral'

class TextureAnalyzer:
    def __init__(self):
        pass
        
    def analyze_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture properties of the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using LBP and other methods
        # Local Binary Pattern
        lbp = self.calculate_lbp(gray)
        lbp_var = np.var(lbp)
        
        # Gabor filter responses
        gabor_responses = self.calculate_gabor_responses(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        return {
            'lbp_variance': float(lbp_var),
            'gabor_energy': float(np.mean(gabor_responses)),
            'edge_density': float(edge_density),
            'gradient_magnitude': float(gradient_mean)
        }
    
    def calculate_lbp(self, image: np.ndarray, radius=1, n_points=8) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        # Simplified LBP implementation
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if image[x, y] >= center:
                        code += 2**k
                lbp[i, j] = code
        return lbp
    
    def calculate_gabor_responses(self, image: np.ndarray) -> np.ndarray:
        """Calculate Gabor filter responses"""
        responses = []
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 
                                       2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            responses.append(np.mean(response))
        return np.array(responses)

class SceneAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize models
        self.scene_classifier = SceneClassifier(num_classes=365).to(device)
        self.lighting_classifier = LightingClassifier().to(device)
        self.indoor_outdoor_classifier = IndoorOutdoorClassifier().to(device)
        
        # Initialize analyzers
        self.color_analyzer = ColorAnalyzer()
        self.lighting_analyzer = LightingAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Scene categories (Places365)
        self.scene_categories = [
            'bedroom', 'living_room', 'kitchen', 'bathroom', 'office',
            'outdoor', 'street', 'park', 'beach', 'forest', 'mountain',
            'restaurant', 'shop', 'church', 'hospital', 'school'
        ]  # Simplified list
        
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive scene analysis"""
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Scene classification
        scene_result = self.classify_scene(input_tensor)
        
        # Indoor/Outdoor classification
        indoor_outdoor = self.classify_indoor_outdoor(input_tensor)
        
        # Lighting analysis
        lighting_analysis = self.lighting_analyzer.analyze_lighting(image)
        
        # Color analysis
        dominant_colors = self.color_analyzer.extract_dominant_colors(image)
        color_harmony = self.color_analyzer.analyze_color_harmony(dominant_colors)
        
        # Texture analysis
        texture_analysis = self.texture_analyzer.analyze_texture(image)
        
        # Environment type inference
        environment_type = self.infer_environment_type(scene_result, indoor_outdoor, lighting_analysis)
        
        return {
            'scene_category': scene_result['category'],
            'scene_confidence': scene_result['confidence'],
            'indoor_outdoor': indoor_outdoor['prediction'],
            'indoor_outdoor_confidence': indoor_outdoor['confidence'],
            'lighting_condition': self.interpret_lighting_condition(lighting_analysis),
            'dominant_colors': dominant_colors,
            'color_harmony': color_harmony,
            'texture_properties': texture_analysis,
            'environment_type': environment_type
        }
    
    def classify_scene(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Classify scene category"""
        self.scene_classifier.eval()
        with torch.no_grad():
            outputs = self.scene_classifier(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        # Map to category name (simplified)
        category_idx = predicted_class.item() % len(self.scene_categories)
        category = self.scene_categories[category_idx]
        
        return {
            'category': category,
            'confidence': confidence.item(),
            'raw_outputs': outputs.cpu().numpy()
        }
    
    def classify_indoor_outdoor(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Classify indoor vs outdoor"""
        # Extract features from scene classifier
        self.scene_classifier.eval()
        with torch.no_grad():
            # Get features from penultimate layer
            features = self.scene_classifier.backbone.avgpool(
                self.scene_classifier.backbone.layer4(
                    self.scene_classifier.backbone.layer3(
                        self.scene_classifier.backbone.layer2(
                            self.scene_classifier.backbone.layer1(
                                self.scene_classifier.backbone.conv1(input_tensor)
                            )
                        )
                    )
                )
            )
            features = features.view(features.size(0), -1)
            
            # Classify indoor/outdoor
            outputs = self.indoor_outdoor_classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        prediction = 'indoor' if predicted_class.item() == 0 else 'outdoor'
        
        return {
            'prediction': prediction,
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()
        }
    
    def interpret_lighting_condition(self, lighting_analysis: Dict[str, Any]) -> str:
        """Interpret lighting condition from analysis"""
        brightness = lighting_analysis['brightness_mean']
        contrast = lighting_analysis['contrast']
        shadows_ratio = lighting_analysis['shadows_ratio']
        highlights_ratio = lighting_analysis['highlights_ratio']
        color_temp = lighting_analysis['color_temperature']
        
        # Rule-based interpretation
        if brightness < 50:
            return 'low_light'
        elif brightness > 200:
            return 'bright_light'
        elif contrast > 0.8:
            return 'high_contrast'
        elif shadows_ratio > 0.3:
            return 'harsh_shadows'
        elif highlights_ratio > 0.2:
            return 'overexposed'
        elif color_temp == 'warm':
            return 'warm_lighting'
        elif color_temp == 'cool':
            return 'cool_lighting'
        else:
            return 'normal_lighting'
    
    def infer_environment_type(self, scene_result: Dict[str, Any], 
                              indoor_outdoor: Dict[str, Any],
                              lighting_analysis: Dict[str, Any]) -> str:
        """Infer specific environment type"""
        scene_category = scene_result['category']
        is_indoor = indoor_outdoor['prediction'] == 'indoor'
        brightness = lighting_analysis['brightness_mean']
        
        if is_indoor:
            if scene_category in ['bedroom', 'living_room', 'kitchen', 'bathroom']:
                return 'residential_indoor'
            elif scene_category in ['office', 'shop', 'restaurant']:
                return 'commercial_indoor'
            elif scene_category in ['hospital', 'school', 'church']:
                return 'institutional_indoor'
            else:
                return 'indoor_generic'
        else:
            if scene_category in ['street', 'shop']:
                return 'urban_outdoor'
            elif scene_category in ['park', 'beach', 'forest', 'mountain']:
                return 'natural_outdoor'
            else:
                return 'outdoor_generic'

# Training configuration
SCENE_ANALYSIS_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'weight_decay': 1e-5,
    'scene_classes': 365,
    'lighting_conditions': 8,
    'augmentation': {
        'rotation': 15,
        'brightness': 0.2,
        'contrast': 0.2,
        'hue': 0.1
    }
}

# Evaluation metrics
def evaluate_scene_analysis(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Evaluate scene analysis results"""
    scene_correct = 0
    indoor_outdoor_correct = 0
    lighting_correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        if pred['scene_category'] == gt['scene_category']:
            scene_correct += 1
        if pred['indoor_outdoor'] == gt['indoor_outdoor']:
            indoor_outdoor_correct += 1
        if pred['lighting_condition'] == gt['lighting_condition']:
            lighting_correct += 1
            
    return {
        'scene_accuracy': scene_correct / total,
        'indoor_outdoor_accuracy': indoor_outdoor_correct / total,
        'lighting_accuracy': lighting_correct / total,
        'overall_accuracy': (scene_correct + indoor_outdoor_correct + lighting_correct) / (3 * total)
    }