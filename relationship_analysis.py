import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import cv2

class SceneGraphNode:
    def __init__(self, object_id: int, bbox: Tuple[float, float, float, float], 
                 class_name: str, features: torch.Tensor):
        self.object_id = object_id
        self.bbox = bbox
        self.class_name = class_name
        self.features = features
        self.spatial_features = self.compute_spatial_features()
        
    def compute_spatial_features(self):
        """Compute spatial features from bounding box"""
        x1, y1, x2, y2 = self.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        return torch.tensor([center_x, center_y, width, height, area], dtype=torch.float32)

class SceneGraphEdge:
    def __init__(self, subject_id: int, object_id: int, relationship: str, 
                 confidence: float, spatial_relation: torch.Tensor):
        self.subject_id = subject_id
        self.object_id = object_id
        self.relationship = relationship
        self.confidence = confidence
        self.spatial_relation = spatial_relation

class SpatialRelationEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=32):
        super(SpatialRelationEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, spatial_features):
        return self.encoder(spatial_features)

class RelationshipClassifier(nn.Module):
    def __init__(self, object_feature_dim=256, spatial_feature_dim=32, 
                 hidden_dim=128, num_relationships=50):
        super(RelationshipClassifier, self).__init__()
        
        # Object feature processing
        self.object_encoder = nn.Sequential(
            nn.Linear(object_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Spatial relation encoder
        self.spatial_encoder = SpatialRelationEncoder(
            input_dim=10, hidden_dim=64, output_dim=spatial_feature_dim
        )
        
        # Relationship classifier
        self.relationship_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + spatial_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_relationships)
        )
        
    def forward(self, subject_features, object_features, spatial_features):
        # Encode object features
        subject_encoded = self.object_encoder(subject_features)
        object_encoded = self.object_encoder(object_features)
        
        # Encode spatial features
        spatial_encoded = self.spatial_encoder(spatial_features)
        
        # Concatenate all features
        combined_features = torch.cat([subject_encoded, object_encoded, spatial_encoded], dim=-1)
        
        # Classify relationship
        relationship_logits = self.relationship_classifier(combined_features)
        
        return relationship_logits

class GraphNeuralNetwork(nn.Module):
    def __init__(self, node_feature_dim=256, edge_feature_dim=32, 
                 hidden_dim=128, num_layers=3):
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(node_feature_dim, hidden_dim, heads=4, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # Output layer
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                
        return x

class RelationshipAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        self.relationship_classifier = RelationshipClassifier().to(device)
        self.graph_network = GraphNeuralNetwork().to(device)
        
        # Predefined spatial relationships
        self.spatial_relationships = [
            'left_of', 'right_of', 'above', 'below', 'inside', 'outside',
            'on', 'under', 'near', 'far_from', 'touching', 'overlapping'
        ]
        
        # Semantic relationships
        self.semantic_relationships = [
            'holding', 'wearing', 'sitting_on', 'standing_on', 'lying_on',
            'driving', 'riding', 'eating', 'drinking', 'playing_with',
            'looking_at', 'talking_to', 'walking_with', 'carrying'
        ]
        
        self.all_relationships = self.spatial_relationships + self.semantic_relationships
        
    def analyze_relationships(self, image: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """Analyze relationships between detected objects"""
        if len(objects) < 2:
            return []
            
        # Create scene graph nodes
        nodes = self.create_scene_graph_nodes(objects)
        
        # Generate all possible pairs
        relationships = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    relationship = self.predict_relationship(nodes[i], nodes[j], image)
                    if relationship['confidence'] > 0.3:  # Threshold
                        relationships.append(relationship)
                        
        # Apply graph neural network for refinement
        refined_relationships = self.refine_with_gnn(nodes, relationships)
        
        return refined_relationships
    
    def create_scene_graph_nodes(self, objects: List[Dict]) -> List[SceneGraphNode]:
        """Create scene graph nodes from detected objects"""
        nodes = []
        for i, obj in enumerate(objects):
            # Extract features (placeholder - in practice, use CNN features)
            features = torch.randn(256).to(self.device)  # Placeholder
            
            node = SceneGraphNode(
                object_id=i,
                bbox=obj['bbox'],
                class_name=obj.get('class_name', 'unknown'),
                features=features
            )
            nodes.append(node)
            
        return nodes
    
    def predict_relationship(self, subject: SceneGraphNode, object_node: SceneGraphNode, 
                           image: np.ndarray) -> Dict:
        """Predict relationship between two objects"""
        # Compute spatial features
        spatial_features = self.compute_spatial_relationship_features(subject, object_node)
        
        # Predict relationship using classifier
        with torch.no_grad():
            relationship_logits = self.relationship_classifier(
                subject.features.unsqueeze(0),
                object_node.features.unsqueeze(0),
                spatial_features.unsqueeze(0)
            )
            
        # Get prediction
        probabilities = F.softmax(relationship_logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)
        
        relationship_name = self.all_relationships[predicted_class.item()]
        
        return {
            'subject_id': subject.object_id,
            'object_id': object_node.object_id,
            'relationship': relationship_name,
            'confidence': confidence.item(),
            'spatial_features': spatial_features.cpu().numpy()
        }
    
    def compute_spatial_relationship_features(self, subject: SceneGraphNode, 
                                            object_node: SceneGraphNode) -> torch.Tensor:
        """Compute spatial relationship features between two objects"""
        subj_x1, subj_y1, subj_x2, subj_y2 = subject.bbox
        obj_x1, obj_y1, obj_x2, obj_y2 = object_node.bbox
        
        # Centers
        subj_center_x = (subj_x1 + subj_x2) / 2
        subj_center_y = (subj_y1 + subj_y2) / 2
        obj_center_x = (obj_x1 + obj_x2) / 2
        obj_center_y = (obj_y1 + obj_y2) / 2
        
        # Relative position
        rel_x = obj_center_x - subj_center_x
        rel_y = obj_center_y - subj_center_y
        
        # Distance
        distance = np.sqrt(rel_x**2 + rel_y**2)
        
        # Angle
        angle = np.arctan2(rel_y, rel_x)
        
        # Size ratio
        subj_area = (subj_x2 - subj_x1) * (subj_y2 - subj_y1)
        obj_area = (obj_x2 - obj_x1) * (obj_y2 - obj_y1)
        size_ratio = obj_area / (subj_area + 1e-6)
        
        # Overlap
        overlap_x = max(0, min(subj_x2, obj_x2) - max(subj_x1, obj_x1))
        overlap_y = max(0, min(subj_y2, obj_y2) - max(subj_y1, obj_y1))
        overlap_area = overlap_x * overlap_y
        overlap_ratio = overlap_area / (subj_area + obj_area - overlap_area + 1e-6)
        
        features = torch.tensor([
            rel_x, rel_y, distance, angle, size_ratio,
            overlap_ratio, subj_center_x, subj_center_y,
            obj_center_x, obj_center_y
        ], dtype=torch.float32).to(self.device)
        
        return features
    
    def refine_with_gnn(self, nodes: List[SceneGraphNode], 
                       relationships: List[Dict]) -> List[Dict]:
        """Refine relationships using Graph Neural Network"""
        if len(nodes) < 2 or len(relationships) == 0:
            return relationships
            
        # Create graph data
        node_features = torch.stack([node.features for node in nodes])
        edge_index = []
        edge_features = []
        
        for rel in relationships:
            edge_index.append([rel['subject_id'], rel['object_id']])
            edge_features.append(torch.tensor(rel['spatial_features'], dtype=torch.float32))
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        edge_features = torch.stack(edge_features).to(self.device)
        
        # Create graph data
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        
        # Apply GNN
        with torch.no_grad():
            refined_features = self.graph_network(
                graph_data.x, graph_data.edge_index, graph_data.edge_attr
            )
            
        # Update relationship confidences based on refined features
        refined_relationships = []
        for i, rel in enumerate(relationships):
            # Compute new confidence based on refined features
            subject_refined = refined_features[rel['subject_id']]
            object_refined = refined_features[rel['object_id']]
            
            # Simple confidence update (in practice, use learned mapping)
            feature_similarity = F.cosine_similarity(
                subject_refined.unsqueeze(0), object_refined.unsqueeze(0)
            )
            
            new_confidence = (rel['confidence'] + feature_similarity.item()) / 2
            
            refined_rel = rel.copy()
            refined_rel['confidence'] = new_confidence
            refined_relationships.append(refined_rel)
            
        return refined_relationships

class RelationshipTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader, optimizer):
        """Train relationship classifier for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            subject_features = batch['subject_features'].to(self.device)
            object_features = batch['object_features'].to(self.device)
            spatial_features = batch['spatial_features'].to(self.device)
            relationship_labels = batch['relationship_labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(subject_features, object_features, spatial_features)
            
            # Calculate loss
            loss = self.criterion(logits, relationship_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

# Training configuration for relationship analysis
RELATIONSHIP_TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'weight_decay': 1e-5,
    'hidden_dim': 128,
    'spatial_feature_dim': 32,
    'num_relationships': 26  # spatial + semantic
}

# Evaluation metrics
def evaluate_relationships(predictions, ground_truth, threshold=0.5):
    """Evaluate relationship predictions"""
    correct = 0
    total = 0
    
    for pred, gt in zip(predictions, ground_truth):
        if pred['confidence'] > threshold:
            if (pred['subject_id'] == gt['subject_id'] and 
                pred['object_id'] == gt['object_id'] and
                pred['relationship'] == gt['relationship']):
                correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }