import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.cluster import DBSCAN
import skimage.feature as feature
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class MaterialClassifier(nn.Module):
    """Material classification network"""
    def __init__(self, num_materials=23):
        super(MaterialClassifier, self).__init__()
        
        # ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Material classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_materials)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        materials = self.classifier(features)
        return materials

class TextureAnalyzer(nn.Module):
    """Deep texture analysis network"""
    def __init__(self, num_texture_classes=47):
        super(TextureAnalyzer, self).__init__()
        
        # EfficientNet backbone
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Multi-scale texture features
        self.texture_branch = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Texture classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_texture_classes)
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        texture_features = self.texture_branch(features)
        texture_class = self.classifier(texture_features)
        return texture_class, texture_features

class ShadowDetector(nn.Module):
    """Shadow detection and analysis"""
    def __init__(self):
        super(ShadowDetector, self).__init__()
        
        # U-Net architecture for shadow detection
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        shadow_mask = self.decoder(encoded)
        return shadow_mask

class OcclusionAnalyzer(nn.Module):
    """Occlusion detection and analysis"""
    def __init__(self):
        super(OcclusionAnalyzer, self).__init__()
        
        # ResNet18 for occlusion detection
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Occlusion classifier
        self.occlusion_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # No occlusion, partial, heavy
        )
        
        # Occlusion mask generator
        self.mask_generator = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        
        # Global features for classification
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        occlusion_class = self.occlusion_classifier(global_features)
        
        # Occlusion mask
        occlusion_mask = self.mask_generator(features)
        
        return occlusion_class, occlusion_mask

class LightingAnalyzer(nn.Module):
    """Detailed lighting analysis"""
    def __init__(self):
        super(LightingAnalyzer, self).__init__()
        
        # ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Lighting direction estimation
        self.direction_estimator = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # 3D lighting direction
        )
        
        # Lighting intensity estimation
        self.intensity_estimator = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # Lighting type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)  # natural, fluorescent, incandescent, LED, mixed
        )
        
    def forward(self, x):
        features = self.backbone(x)
        
        direction = self.direction_estimator(features)
        intensity = self.intensity_estimator(features)
        lighting_type = self.type_classifier(features)
        
        return direction, intensity, lighting_type

class AffordanceDetector(nn.Module):
    """Object affordance detection"""
    def __init__(self, num_affordances=15):
        super(AffordanceDetector, self).__init__()
        
        # ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Affordance classifier
        self.affordance_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_affordances)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        affordances = self.affordance_classifier(features)
        return affordances

class FineDetailExtractor:
    """Main class for fine-grained visual detail extraction"""
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize models
        self.material_classifier = MaterialClassifier().to(device)
        self.texture_analyzer = TextureAnalyzer().to(device)
        self.shadow_detector = ShadowDetector().to(device)
        self.occlusion_analyzer = OcclusionAnalyzer().to(device)
        self.lighting_analyzer = LightingAnalyzer().to(device)
        self.affordance_detector = AffordanceDetector().to(device)
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Material labels
        self.material_labels = [
            'fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic',
            'stone', 'water', 'wood', 'ceramic', 'concrete', 'fur', 'hair',
            'painted', 'polished', 'rubber', 'skin', 'soil', 'brick', 'carpet',
            'tile', 'asphalt'
        ]
        
        # Texture labels
        self.texture_labels = [
            'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered',
            'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted',
            'fibrous', 'flecked', 'flowery', 'frilly', 'gauzy', 'grid',
            'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike',
            'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated',
            'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed',
            'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained',
            'stratified', 'striped', 'studded', 'swirly', 'veined',
            'waffled', 'woven', 'wrinkled', 'zigzagged'
        ]
        
        # Affordance labels
        self.affordance_labels = [
            'graspable', 'cuttable', 'drinkable', 'edible', 'readable',
            'sittable', 'walkable', 'pushable', 'pullable', 'liftable',
            'openable', 'closable', 'wearable', 'throwable', 'breakable'
        ]
        
    def extract_details(self, image: np.ndarray, objects: List[Dict]) -> Dict[str, Any]:
        """Extract fine-grained visual details from image"""
        # Global image analysis
        global_details = self.analyze_global_image(image)
        
        # Object-specific analysis
        object_details = []
        for obj in objects:
            obj_details = self.analyze_object_region(image, obj)
            object_details.append(obj_details)
            
        # Texture analysis
        texture_details = self.analyze_texture_patterns(image)
        
        # Shadow analysis
        shadow_details = self.analyze_shadows(image)
        
        # Occlusion analysis
        occlusion_details = self.analyze_occlusions(image)
        
        # Lighting analysis
        lighting_details = self.analyze_lighting_conditions(image)
        
        return {
            'global_details': global_details,
            'object_details': object_details,
            'texture_details': texture_details,
            'shadow_details': shadow_details,
            'occlusion_details': occlusion_details,
            'lighting_details': lighting_details
        }
    
    def analyze_global_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze global image properties"""
        # Convert to different color spaces
        lab_image = rgb2lab(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate global statistics
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        contrast = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # Color distribution
        color_hist = self.calculate_color_histogram(image)
        
        # Edge density
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture energy
        texture_energy = self.calculate_texture_energy(image)
        
        # Symmetry analysis
        symmetry_score = self.calculate_symmetry_score(image)
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'texture_energy': float(texture_energy),
            'symmetry_score': float(symmetry_score),
            'color_histogram': color_hist
        }
    
    def analyze_object_region(self, image: np.ndarray, obj: Dict) -> Dict[str, Any]:
        """Analyze specific object region"""
        # Extract object region
        x1, y1, x2, y2 = obj['bbox']
        object_region = image[int(y1):int(y2), int(x1):int(x2)]
        
        if object_region.size == 0:
            return {}
        
        # Resize for analysis
        object_region = cv2.resize(object_region, (224, 224))
        
        # Material classification
        material_result = self.classify_material(object_region)
        
        # Texture analysis
        texture_result = self.analyze_object_texture(object_region)
        
        # Affordance detection
        affordance_result = self.detect_affordances(object_region)
        
        # Surface properties
        surface_properties = self.analyze_surface_properties(object_region)
        
        return {
            'material': material_result,
            'texture': texture_result,
            'affordances': affordance_result,
            'surface_properties': surface_properties
        }
    
    def classify_material(self, object_region: np.ndarray) -> Dict[str, Any]:
        """Classify material of object region"""
        # Preprocess
        input_tensor = self.transform(object_region).unsqueeze(0).to(self.device)
        
        # Classify material
        with torch.no_grad():
            material_logits = self.material_classifier(input_tensor)
            material_probs = F.softmax(material_logits, dim=1)
            confidence, predicted_class = torch.max(material_probs, dim=1)
            
        material_name = self.material_labels[predicted_class.item()]
        
        return {
            'material': material_name,
            'confidence': confidence.item(),
            'all_probabilities': {
                self.material_labels[i]: material_probs[0, i].item()
                for i in range(len(self.material_labels))
            }
        }
    
    def analyze_object_texture(self, object_region: np.ndarray) -> Dict[str, Any]:
        """Analyze texture of object region"""
        # Preprocess
        input_tensor = self.transform(object_region).unsqueeze(0).to(self.device)
        
        # Analyze texture
        with torch.no_grad():
            texture_logits, texture_features = self.texture_analyzer(input_tensor)
            texture_probs = F.softmax(texture_logits, dim=1)
            confidence, predicted_class = torch.max(texture_probs, dim=1)
            
        texture_name = self.texture_labels[predicted_class.item()]
        
        # Additional texture analysis
        texture_stats = self.calculate_texture_statistics(object_region)
        
        return {
            'texture_class': texture_name,
            'confidence': confidence.item(),
            'texture_features': texture_features.cpu().numpy(),
            'texture_statistics': texture_stats
        }
    
    def detect_affordances(self, object_region: np.ndarray) -> Dict[str, Any]:
        """Detect object affordances"""
        # Preprocess
        input_tensor = self.transform(object_region).unsqueeze(0).to(self.device)
        
        # Detect affordances
        with torch.no_grad():
            affordance_logits = self.affordance_detector(input_tensor)
            affordance_probs = torch.sigmoid(affordance_logits)
            
        # Multi-label classification
        affordances = {}
        for i, affordance in enumerate(self.affordance_labels):
            affordances[affordance] = affordance_probs[0, i].item()
            
        return affordances
    
    def analyze_surface_properties(self, object_region: np.ndarray) -> Dict[str, Any]:
        """Analyze surface properties"""
        gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
        
        # Roughness estimation
        roughness = self.estimate_roughness(gray)
        
        # Reflectance estimation
        reflectance = self.estimate_reflectance(object_region)
        
        # Glossiness estimation
        glossiness = self.estimate_glossiness(gray)
        
        # Transparency estimation
        transparency = self.estimate_transparency(object_region)
        
        return {
            'roughness': float(roughness),
            'reflectance': float(reflectance),
            'glossiness': float(glossiness),
            'transparency': float(transparency)
        }
    
    def analyze_texture_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture patterns in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp, bins=10)[0]
        
        # Gray-Level Co-occurrence Matrix
        glcm = feature.greycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
        
        # Texture properties from GLCM
        contrast = feature.greycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = feature.greycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = feature.greycoprops(glcm, 'homogeneity')[0, 0]
        energy = feature.greycoprops(glcm, 'energy')[0, 0]
        
        # Gabor filter responses
        gabor_responses = []
        for theta in range(0, 180, 30):
            filt_real, _ = feature.gabor_filter(gray, frequency=0.6, theta=np.deg2rad(theta))
            gabor_responses.append(np.var(filt_real))
            
        return {
            'lbp_histogram': lbp_hist.tolist(),
            'glcm_contrast': float(contrast),
            'glcm_dissimilarity': float(dissimilarity),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy),
            'gabor_responses': gabor_responses
        }
    
    def analyze_shadows(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze shadows in the image"""
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Detect shadows
        with torch.no_grad():
            shadow_mask = self.shadow_detector(input_tensor)
            
        shadow_mask_np = shadow_mask.squeeze().cpu().numpy()
        
        # Shadow statistics
        shadow_ratio = np.mean(shadow_mask_np)
        shadow_intensity = np.mean(shadow_mask_np[shadow_mask_np > 0.5])
        
        # Shadow regions
        shadow_regions = self.find_shadow_regions(shadow_mask_np)
        
        return {
            'shadow_ratio': float(shadow_ratio),
            'shadow_intensity': float(shadow_intensity),
            'num_shadow_regions': len(shadow_regions),
            'shadow_mask': shadow_mask_np
        }
    
    def analyze_occlusions(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze occlusions in the image"""
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Detect occlusions
        with torch.no_grad():
            occlusion_class, occlusion_mask = self.occlusion_analyzer(input_tensor)
            
        occlusion_probs = F.softmax(occlusion_class, dim=1)
        occlusion_type = ['none', 'partial', 'heavy'][torch.argmax(occlusion_probs).item()]
        
        occlusion_mask_np = occlusion_mask.squeeze().cpu().numpy()
        
        return {
            'occlusion_type': occlusion_type,
            'occlusion_confidence': torch.max(occlusion_probs).item(),
            'occlusion_ratio': float(np.mean(occlusion_mask_np)),
            'occlusion_mask': occlusion_mask_np
        }
    
    def analyze_lighting_conditions(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze detailed lighting conditions"""
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Analyze lighting
        with torch.no_grad():
            direction, intensity, lighting_type = self.lighting_analyzer(input_tensor)
            
        lighting_type_probs = F.softmax(lighting_type, dim=1)
        lighting_types = ['natural', 'fluorescent', 'incandescent', 'LED', 'mixed']
        predicted_type = lighting_types[torch.argmax(lighting_type_probs).item()]
        
        # Normalize direction vector
        direction_norm = F.normalize(direction, dim=1)
        
        return {
            'lighting_direction': direction_norm.squeeze().cpu().numpy().tolist(),
            'lighting_intensity': intensity.squeeze().cpu().numpy().item(),
            'lighting_type': predicted_type,
            'lighting_type_confidence': torch.max(lighting_type_probs).item()
        }
    
    def calculate_color_histogram(self, image: np.ndarray) -> Dict[str, List]:
        """Calculate color histogram"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        return {
            'hue': h_hist.flatten().tolist(),
            'saturation': s_hist.flatten().tolist(),
            'value': v_hist.flatten().tolist()
        }
    
    def calculate_texture_energy(self, image: np.ndarray) -> float:
        """Calculate texture energy"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(gradient_magnitude)
    
    def calculate_symmetry_score(self, image: np.ndarray) -> float:
        """Calculate symmetry score"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Vertical symmetry
        left_half = gray[:, :gray.shape[1]//2]
        right_half = gray[:, gray.shape[1]//2:]
        right_half_flipped = np.fliplr(right_half)
        
        # Resize to match dimensions
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Calculate similarity
        symmetry = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
        
        return symmetry if not np.isnan(symmetry) else 0.0
    
    def calculate_texture_statistics(self, region: np.ndarray) -> Dict[str, float]:
        """Calculate texture statistics"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Statistical measures
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        skewness = self.calculate_skewness(gray)
        kurtosis = self.calculate_kurtosis(gray)
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }
    
    def calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def estimate_roughness(self, gray: np.ndarray) -> float:
        """Estimate surface roughness"""
        # Use Laplacian variance as roughness measure
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        roughness = np.var(laplacian)
        return roughness
    
    def estimate_reflectance(self, region: np.ndarray) -> float:
        """Estimate surface reflectance"""
        # Use brightness as a proxy for reflectance
        brightness = np.mean(region)
        return brightness / 255.0
    
    def estimate_glossiness(self, gray: np.ndarray) -> float:
        """Estimate surface glossiness"""
        # Use gradient magnitude variance as glossiness measure
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        glossiness = np.var(gradient_magnitude)
        return glossiness
    
    def estimate_transparency(self, region: np.ndarray) -> float:
        """Estimate surface transparency"""
        # Use alpha channel if available, otherwise use edge sharpness
        if region.shape[2] == 4:  # RGBA
            return np.mean(region[:, :, 3]) / 255.0
        else:
            # Use edge sharpness as proxy
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return 1.0 - edge_density  # Higher edge density = less transparency
    
    def find_shadow_regions(self, shadow_mask: np.ndarray) -> List[Dict]:
        """Find shadow regions in the image"""
        # Threshold shadow mask
        binary_mask = (shadow_mask > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'contour': contour.tolist()
                })
                
        return regions

# Training configuration
FINE_DETAIL_CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'weight_decay': 1e-5,
    'materials': 23,
    'textures': 47,
    'affordances': 15,
    'augmentation': {
        'rotation': 30,
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.2
    }
}

# Evaluation metrics
def evaluate_fine_details(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Evaluate fine detail extraction results"""
    material_correct = 0
    texture_correct = 0
    lighting_error = 0
    shadow_iou = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        # Material accuracy
        if pred['material']['material'] == gt['material']:
            material_correct += 1
            
        # Texture accuracy
        if pred['texture']['texture_class'] == gt['texture']:
            texture_correct += 1
            
        # Lighting direction error
        pred_dir = np.array(pred['lighting_details']['lighting_direction'])
        gt_dir = np.array(gt['lighting_direction'])
        lighting_error += np.linalg.norm(pred_dir - gt_dir)
        
        # Shadow IoU
        pred_shadow = pred['shadow_details']['shadow_mask']
        gt_shadow = gt['shadow_mask']
        intersection = np.logical_and(pred_shadow > 0.5, gt_shadow > 0.5)
        union = np.logical_or(pred_shadow > 0.5, gt_shadow > 0.5)
        shadow_iou += np.sum(intersection) / np.sum(union)
    
    return {
        'material_accuracy': material_correct / total,
        'texture_accuracy': texture_correct / total,
        'lighting_mae': lighting_error / total,
        'shadow_iou': shadow_iou / total
    }