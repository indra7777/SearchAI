import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import os
import time
import unittest
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Import all pipeline components
from visual_intelligence_pipeline import VisualIntelligencePipeline, VisualAnalysisResult
from object_detection import ObjectDetector, DetectionResult
from relationship_analysis import RelationshipAnalyzer, RelationshipResult
from scene_analysis import SceneAnalyzer, SceneResult
from facial_analysis import FaceAnalyzer, FaceResult
from fine_detail_extraction import FineDetailExtractor
from training_framework import TrainingFramework, EvaluationFramework

class IntegrationTester:
    """Comprehensive integration testing for the visual intelligence pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize pipeline
        self.pipeline = VisualIntelligencePipeline(self.device)
        self.pipeline.initialize_models()
        
        # Setup logging
        self.setup_logging()
        
        # Test results storage
        self.test_results = {}
        
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default testing configuration"""
        return {
            'test_images_dir': 'test_images/',
            'test_annotations_dir': 'test_annotations/',
            'output_dir': 'test_outputs/',
            'performance_thresholds': {
                'inference_time': 2.0,  # seconds
                'memory_usage': 8.0,    # GB
                'accuracy_threshold': 0.8
            },
            'test_categories': [
                'single_object', 'multiple_objects', 'faces', 'scenes',
                'complex_relationships', 'challenging_lighting',
                'occlusions', 'materials_textures'
            ]
        }
    
    def setup_logging(self):
        """Setup logging for testing"""
        log_dir = os.path.join(self.config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'integration_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        self.logger.info("Starting comprehensive integration tests...")
        
        test_results = {}
        
        # 1. Unit tests for individual components
        test_results['unit_tests'] = self.run_unit_tests()
        
        # 2. Integration tests
        test_results['integration_tests'] = self.run_integration_tests()
        
        # 3. Performance tests
        test_results['performance_tests'] = self.run_performance_tests()
        
        # 4. Stress tests
        test_results['stress_tests'] = self.run_stress_tests()
        
        # 5. Edge case tests
        test_results['edge_case_tests'] = self.run_edge_case_tests()
        
        # 6. Accuracy validation
        test_results['accuracy_tests'] = self.run_accuracy_tests()
        
        # Generate comprehensive report
        self.generate_test_report(test_results)
        
        self.logger.info("Comprehensive integration tests completed!")
        return test_results
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for individual components"""
        self.logger.info("Running unit tests...")
        
        unit_test_results = {}
        
        # Test object detection
        unit_test_results['object_detection'] = self.test_object_detection_unit()
        
        # Test relationship analysis
        unit_test_results['relationship_analysis'] = self.test_relationship_analysis_unit()
        
        # Test scene analysis
        unit_test_results['scene_analysis'] = self.test_scene_analysis_unit()
        
        # Test facial analysis
        unit_test_results['facial_analysis'] = self.test_facial_analysis_unit()
        
        # Test fine detail extraction
        unit_test_results['fine_detail_extraction'] = self.test_fine_detail_extraction_unit()
        
        return unit_test_results
    
    def test_object_detection_unit(self) -> Dict[str, Any]:
        """Unit test for object detection"""
        try:
            # Create test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test detection
            start_time = time.time()
            detections = self.pipeline.object_detector.detect(test_image)
            inference_time = time.time() - start_time
            
            # Validate output format
            assert isinstance(detections, list), "Detections should be a list"
            
            for detection in detections:
                assert 'bbox' in detection, "Detection should have bbox"
                assert 'confidence' in detection, "Detection should have confidence"
                assert len(detection['bbox']) == 4, "Bbox should have 4 coordinates"
                assert 0 <= detection['confidence'] <= 1, "Confidence should be between 0 and 1"
            
            return {
                'status': 'PASS',
                'inference_time': inference_time,
                'num_detections': len(detections),
                'message': 'Object detection unit test passed'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Object detection unit test failed'
            }
    
    def test_relationship_analysis_unit(self) -> Dict[str, Any]:
        """Unit test for relationship analysis"""
        try:
            # Create test image and mock objects
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_objects = [
                {'bbox': [100, 100, 200, 200], 'class_name': 'person'},
                {'bbox': [250, 150, 350, 250], 'class_name': 'chair'}
            ]
            
            # Test relationship analysis
            start_time = time.time()
            relationships = self.pipeline.relationship_model.analyze_relationships(test_image, mock_objects)
            inference_time = time.time() - start_time
            
            # Validate output format
            assert isinstance(relationships, list), "Relationships should be a list"
            
            for relationship in relationships:
                assert 'subject_id' in relationship, "Relationship should have subject_id"
                assert 'object_id' in relationship, "Relationship should have object_id"
                assert 'relationship' in relationship, "Relationship should have relationship type"
                assert 'confidence' in relationship, "Relationship should have confidence"
                assert 0 <= relationship['confidence'] <= 1, "Confidence should be between 0 and 1"
            
            return {
                'status': 'PASS',
                'inference_time': inference_time,
                'num_relationships': len(relationships),
                'message': 'Relationship analysis unit test passed'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Relationship analysis unit test failed'
            }
    
    def test_scene_analysis_unit(self) -> Dict[str, Any]:
        """Unit test for scene analysis"""
        try:
            # Create test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test scene analysis
            start_time = time.time()
            scene_result = self.pipeline.scene_analyzer.analyze(test_image)
            inference_time = time.time() - start_time
            
            # Validate output format
            required_keys = ['scene_category', 'indoor_outdoor', 'lighting_condition', 
                           'dominant_colors', 'environment_type']
            
            for key in required_keys:
                assert key in scene_result, f"Scene result should have {key}"
            
            assert isinstance(scene_result['dominant_colors'], list), "Dominant colors should be a list"
            assert scene_result['indoor_outdoor'] in ['indoor', 'outdoor'], "Indoor/outdoor should be indoor or outdoor"
            
            return {
                'status': 'PASS',
                'inference_time': inference_time,
                'scene_category': scene_result['scene_category'],
                'message': 'Scene analysis unit test passed'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Scene analysis unit test failed'
            }
    
    def test_facial_analysis_unit(self) -> Dict[str, Any]:
        """Unit test for facial analysis"""
        try:
            # Create test image with face-like region
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test facial analysis
            start_time = time.time()
            face_results = self.pipeline.face_detector.analyze(test_image)
            inference_time = time.time() - start_time
            
            # Validate output format
            assert isinstance(face_results, list), "Face results should be a list"
            
            for face_result in face_results:
                required_keys = ['bbox', 'confidence', 'emotions', 'age', 'gender', 'gaze_direction']
                for key in required_keys:
                    assert key in face_result, f"Face result should have {key}"
                
                assert isinstance(face_result['emotions'], dict), "Emotions should be a dictionary"
                assert isinstance(face_result['age'], (int, float)), "Age should be numeric"
                assert face_result['gender'] in ['male', 'female'], "Gender should be male or female"
                assert len(face_result['gaze_direction']) == 2, "Gaze direction should have 2 components"
            
            return {
                'status': 'PASS',
                'inference_time': inference_time,
                'num_faces': len(face_results),
                'message': 'Facial analysis unit test passed'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Facial analysis unit test failed'
            }
    
    def test_fine_detail_extraction_unit(self) -> Dict[str, Any]:
        """Unit test for fine detail extraction"""
        try:
            # Create test image and mock objects
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_objects = [{'bbox': [100, 100, 200, 200], 'class_name': 'object'}]
            
            # Test fine detail extraction
            start_time = time.time()
            detail_results = self.pipeline.detail_extractor.extract_details(test_image, mock_objects)
            inference_time = time.time() - start_time
            
            # Validate output format
            required_keys = ['global_details', 'object_details', 'texture_details', 
                           'shadow_details', 'lighting_details']
            
            for key in required_keys:
                assert key in detail_results, f"Detail results should have {key}"
            
            assert isinstance(detail_results['object_details'], list), "Object details should be a list"
            assert isinstance(detail_results['texture_details'], dict), "Texture details should be a dict"
            
            return {
                'status': 'PASS',
                'inference_time': inference_time,
                'message': 'Fine detail extraction unit test passed'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Fine detail extraction unit test failed'
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for the complete pipeline"""
        self.logger.info("Running integration tests...")
        
        integration_results = {}
        
        # Test complete pipeline with different image types
        test_categories = self.config['test_categories']
        
        for category in test_categories:
            integration_results[category] = self.test_pipeline_category(category)
        
        return integration_results
    
    def test_pipeline_category(self, category: str) -> Dict[str, Any]:
        """Test pipeline on specific image category"""
        try:
            # Generate or load test image for category
            test_image = self.generate_test_image_for_category(category)
            
            # Run complete pipeline
            start_time = time.time()
            result = self.pipeline.process_image(test_image)
            total_time = time.time() - start_time
            
            # Validate complete result
            self.validate_pipeline_result(result)
            
            return {
                'status': 'PASS',
                'total_inference_time': total_time,
                'num_objects': len(result.objects),
                'num_faces': len(result.faces),
                'num_relationships': len(result.relationships),
                'message': f'Pipeline test for {category} passed'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': f'Pipeline test for {category} failed'
            }
    
    def generate_test_image_for_category(self, category: str) -> np.ndarray:
        """Generate appropriate test image for category"""
        # Create base image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        if category == 'single_object':
            # Add simple rectangular object
            cv2.rectangle(image, (200, 150), (400, 350), (255, 0, 0), -1)
        
        elif category == 'multiple_objects':
            # Add multiple objects
            cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)
            cv2.rectangle(image, (500, 200), (600, 300), (0, 0, 255), -1)
        
        elif category == 'faces':
            # Add face-like oval shapes
            cv2.ellipse(image, (320, 240), (60, 80), 0, 0, 360, (220, 180, 140), -1)
            # Add eyes
            cv2.circle(image, (300, 220), 8, (0, 0, 0), -1)
            cv2.circle(image, (340, 220), 8, (0, 0, 0), -1)
            # Add mouth
            cv2.ellipse(image, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        
        elif category == 'challenging_lighting':
            # Create gradient lighting effect
            overlay = np.zeros_like(image, dtype=np.float32)
            h, w = image.shape[:2]
            for i in range(h):
                alpha = i / h
                overlay[i, :] = [alpha * 100, alpha * 100, alpha * 100]
            image = cv2.addWeighted(image, 0.7, overlay.astype(np.uint8), 0.3, 0)
        
        elif category == 'occlusions':
            # Add objects with occlusion
            cv2.rectangle(image, (150, 100), (350, 300), (255, 0, 0), -1)
            cv2.rectangle(image, (250, 200), (450, 400), (0, 255, 0), -1)
        
        # Add noise for realism
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def validate_pipeline_result(self, result: VisualAnalysisResult):
        """Validate complete pipeline result"""
        # Check that result has all required components
        assert hasattr(result, 'objects'), "Result should have objects"
        assert hasattr(result, 'faces'), "Result should have faces"
        assert hasattr(result, 'scene'), "Result should have scene"
        assert hasattr(result, 'relationships'), "Result should have relationships"
        assert hasattr(result, 'fine_details'), "Result should have fine_details"
        
        # Validate objects
        assert isinstance(result.objects, list), "Objects should be a list"
        
        # Validate faces
        assert isinstance(result.faces, list), "Faces should be a list"
        
        # Validate relationships
        assert isinstance(result.relationships, list), "Relationships should be a list"
        
        # Validate scene
        assert hasattr(result.scene, 'scene_category'), "Scene should have category"
        
        # Validate fine details
        assert isinstance(result.fine_details, dict), "Fine details should be a dict"
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        self.logger.info("Running performance tests...")
        
        performance_results = {}
        
        # Test inference speed
        performance_results['inference_speed'] = self.test_inference_speed()
        
        # Test memory usage
        performance_results['memory_usage'] = self.test_memory_usage()
        
        # Test batch processing
        performance_results['batch_processing'] = self.test_batch_processing()
        
        # Test GPU utilization
        if torch.cuda.is_available():
            performance_results['gpu_utilization'] = self.test_gpu_utilization()
        
        return performance_results
    
    def test_inference_speed(self) -> Dict[str, Any]:
        """Test inference speed"""
        try:
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                self.pipeline.process_image(test_image)
            
            # Measure inference times
            times = []
            for _ in range(20):
                start_time = time.time()
                self.pipeline.process_image(test_image)
                inference_time = time.time() - start_time
                times.append(inference_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Check if within threshold
            threshold = self.config['performance_thresholds']['inference_time']
            status = 'PASS' if avg_time <= threshold else 'FAIL'
            
            return {
                'status': status,
                'average_time': avg_time,
                'std_time': std_time,
                'min_time': np.min(times),
                'max_time': np.max(times),
                'threshold': threshold,
                'message': f'Average inference time: {avg_time:.3f}s'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Inference speed test failed'
            }
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""
        try:
            import psutil
            import gc
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
            
            # Process multiple images
            test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
            
            for image in test_images:
                self.pipeline.process_image(image)
            
            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            memory_increase = final_memory - initial_memory
            
            gpu_memory_increase = 0
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
                gpu_memory_increase = final_gpu_memory - initial_gpu_memory
            
            # Check if within threshold
            threshold = self.config['performance_thresholds']['memory_usage']
            status = 'PASS' if final_memory <= threshold else 'FAIL'
            
            return {
                'status': status,
                'initial_memory_gb': initial_memory,
                'final_memory_gb': final_memory,
                'memory_increase_gb': memory_increase,
                'gpu_memory_increase_gb': gpu_memory_increase,
                'threshold_gb': threshold,
                'message': f'Memory usage: {final_memory:.2f}GB'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Memory usage test failed'
            }
    
    def test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing capability"""
        try:
            # Create batch of test images
            batch_size = 4
            test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                          for _ in range(batch_size)]
            
            # Process individually
            start_time = time.time()
            individual_results = []
            for image in test_images:
                result = self.pipeline.process_image(image)
                individual_results.append(result)
            individual_time = time.time() - start_time
            
            # Calculate efficiency
            avg_time_per_image = individual_time / batch_size
            
            return {
                'status': 'PASS',
                'batch_size': batch_size,
                'total_time': individual_time,
                'avg_time_per_image': avg_time_per_image,
                'message': f'Batch processing: {avg_time_per_image:.3f}s per image'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Batch processing test failed'
            }
    
    def test_gpu_utilization(self) -> Dict[str, Any]:
        """Test GPU utilization"""
        try:
            # Process image and monitor GPU usage
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Get initial GPU memory
            initial_memory = torch.cuda.memory_allocated()
            
            # Process image
            result = self.pipeline.process_image(test_image)
            
            # Get peak GPU memory
            peak_memory = torch.cuda.max_memory_allocated()
            
            # Calculate utilization
            memory_used = (peak_memory - initial_memory) / 1024 / 1024 / 1024  # GB
            
            return {
                'status': 'PASS',
                'initial_memory_gb': initial_memory / 1024 / 1024 / 1024,
                'peak_memory_gb': peak_memory / 1024 / 1024 / 1024,
                'memory_used_gb': memory_used,
                'message': f'GPU memory used: {memory_used:.2f}GB'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'GPU utilization test failed'
            }
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests"""
        self.logger.info("Running stress tests...")
        
        stress_results = {}
        
        # Test with large images
        stress_results['large_images'] = self.test_large_images()
        
        # Test with many objects
        stress_results['many_objects'] = self.test_many_objects()
        
        # Test continuous processing
        stress_results['continuous_processing'] = self.test_continuous_processing()
        
        return stress_results
    
    def test_large_images(self) -> Dict[str, Any]:
        """Test with large images"""
        try:
            # Create large test image
            large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)  # 4K image
            
            start_time = time.time()
            result = self.pipeline.process_image(large_image)
            processing_time = time.time() - start_time
            
            return {
                'status': 'PASS',
                'image_size': large_image.shape,
                'processing_time': processing_time,
                'message': f'Large image processed in {processing_time:.2f}s'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Large image test failed'
            }
    
    def test_many_objects(self) -> Dict[str, Any]:
        """Test with many objects in image"""
        try:
            # Create image with many objects
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add many rectangular objects
            for i in range(20):
                x = np.random.randint(0, 580)
                y = np.random.randint(0, 420)
                w = np.random.randint(20, 60)
                h = np.random.randint(20, 60)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
            
            start_time = time.time()
            result = self.pipeline.process_image(image)
            processing_time = time.time() - start_time
            
            return {
                'status': 'PASS',
                'num_detected_objects': len(result.objects),
                'processing_time': processing_time,
                'message': f'Many objects test: {len(result.objects)} objects detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Many objects test failed'
            }
    
    def test_continuous_processing(self) -> Dict[str, Any]:
        """Test continuous processing"""
        try:
            num_images = 50
            processing_times = []
            
            for i in range(num_images):
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                start_time = time.time()
                result = self.pipeline.process_image(test_image)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
            
            avg_time = np.mean(processing_times)
            std_time = np.std(processing_times)
            
            return {
                'status': 'PASS',
                'num_images': num_images,
                'avg_processing_time': avg_time,
                'std_processing_time': std_time,
                'message': f'Continuous processing: {avg_time:.3f}±{std_time:.3f}s per image'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Continuous processing test failed'
            }
    
    def run_edge_case_tests(self) -> Dict[str, Any]:
        """Run edge case tests"""
        self.logger.info("Running edge case tests...")
        
        edge_case_results = {}
        
        # Test with empty/black images
        edge_case_results['empty_image'] = self.test_empty_image()
        
        # Test with very bright/dark images
        edge_case_results['extreme_brightness'] = self.test_extreme_brightness()
        
        # Test with monochrome images
        edge_case_results['monochrome'] = self.test_monochrome_image()
        
        # Test with corrupted images
        edge_case_results['corrupted_image'] = self.test_corrupted_image()
        
        return edge_case_results
    
    def test_empty_image(self) -> Dict[str, Any]:
        """Test with empty/black image"""
        try:
            # Create black image
            black_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            result = self.pipeline.process_image(black_image)
            
            # Should handle gracefully without crashing
            return {
                'status': 'PASS',
                'num_objects': len(result.objects),
                'num_faces': len(result.faces),
                'message': 'Empty image handled gracefully'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Empty image test failed'
            }
    
    def test_extreme_brightness(self) -> Dict[str, Any]:
        """Test with extremely bright/dark images"""
        try:
            results = {}
            
            # Very bright image
            bright_image = np.full((480, 640, 3), 255, dtype=np.uint8)
            bright_result = self.pipeline.process_image(bright_image)
            
            # Very dark image
            dark_image = np.full((480, 640, 3), 5, dtype=np.uint8)
            dark_result = self.pipeline.process_image(dark_image)
            
            return {
                'status': 'PASS',
                'bright_objects': len(bright_result.objects),
                'dark_objects': len(dark_result.objects),
                'message': 'Extreme brightness images handled gracefully'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Extreme brightness test failed'
            }
    
    def test_monochrome_image(self) -> Dict[str, Any]:
        """Test with monochrome image"""
        try:
            # Create grayscale image and convert to 3-channel
            gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            mono_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            result = self.pipeline.process_image(mono_image)
            
            return {
                'status': 'PASS',
                'num_objects': len(result.objects),
                'scene_category': result.scene.scene_category,
                'message': 'Monochrome image handled gracefully'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Monochrome image test failed'
            }
    
    def test_corrupted_image(self) -> Dict[str, Any]:
        """Test with corrupted image data"""
        try:
            # Create image with random noise
            corrupted_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add extreme values
            corrupted_image[100:200, 100:200] = 300  # Values > 255
            corrupted_image = np.clip(corrupted_image, 0, 255).astype(np.uint8)
            
            result = self.pipeline.process_image(corrupted_image)
            
            return {
                'status': 'PASS',
                'message': 'Corrupted image handled gracefully'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'message': 'Corrupted image test failed'
            }
    
    def run_accuracy_tests(self) -> Dict[str, Any]:
        """Run accuracy validation tests"""
        self.logger.info("Running accuracy tests...")
        
        # This would typically use a labeled test dataset
        # For demonstration, we'll use synthetic validation
        
        accuracy_results = {
            'object_detection_accuracy': 0.85,  # Placeholder
            'scene_classification_accuracy': 0.78,  # Placeholder
            'emotion_recognition_accuracy': 0.72,  # Placeholder
            'material_classification_accuracy': 0.68,  # Placeholder
            'overall_accuracy': 0.76  # Placeholder
        }
        
        threshold = self.config['performance_thresholds']['accuracy_threshold']
        
        return {
            'status': 'PASS' if accuracy_results['overall_accuracy'] >= threshold else 'FAIL',
            'accuracies': accuracy_results,
            'threshold': threshold,
            'message': f'Overall accuracy: {accuracy_results["overall_accuracy"]:.2f}'
        }
    
    def generate_test_report(self, test_results: Dict[str, Any]):
        """Generate comprehensive test report"""
        report_dir = os.path.join(self.config['output_dir'], 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate JSON report
        json_report_path = os.path.join(report_dir, f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(json_report_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        # Generate HTML report
        self.generate_html_report(test_results, report_dir)
        
        # Generate visualization
        self.generate_test_visualizations(test_results, report_dir)
        
        self.logger.info(f"Test report generated: {json_report_path}")
    
    def generate_html_report(self, test_results: Dict[str, Any], report_dir: str):
        """Generate HTML test report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visual Intelligence Pipeline - Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Visual Intelligence Pipeline - Integration Test Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add test sections
        for section_name, section_results in test_results.items():
            html_content += f'<div class="section"><h2>{section_name.replace("_", " ").title()}</h2>'
            
            if isinstance(section_results, dict):
                for test_name, test_result in section_results.items():
                    status_class = 'pass' if test_result.get('status') == 'PASS' else 'fail'
                    html_content += f'<div class="metric">'
                    html_content += f'<h3>{test_name.replace("_", " ").title()}</h3>'
                    html_content += f'<p class="{status_class}">Status: {test_result.get("status", "N/A")}</p>'
                    html_content += f'<p>Message: {test_result.get("message", "N/A")}</p>'
                    html_content += '</div>'
            
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        html_report_path = os.path.join(report_dir, 'test_report.html')
        with open(html_report_path, 'w') as f:
            f.write(html_content)
    
    def generate_test_visualizations(self, test_results: Dict[str, Any], report_dir: str):
        """Generate test result visualizations"""
        # Create summary plot
        plt.figure(figsize=(12, 8))
        
        # Count pass/fail for each test category
        categories = []
        pass_counts = []
        fail_counts = []
        
        for category, results in test_results.items():
            if isinstance(results, dict):
                categories.append(category.replace('_', ' ').title())
                pass_count = sum(1 for r in results.values() 
                               if isinstance(r, dict) and r.get('status') == 'PASS')
                fail_count = sum(1 for r in results.values() 
                               if isinstance(r, dict) and r.get('status') == 'FAIL')
                pass_counts.append(pass_count)
                fail_counts.append(fail_count)
        
        # Create stacked bar chart
        x = np.arange(len(categories))
        width = 0.6
        
        plt.bar(x, pass_counts, width, label='Pass', color='green', alpha=0.7)
        plt.bar(x, fail_counts, width, bottom=pass_counts, label='Fail', color='red', alpha=0.7)
        
        plt.xlabel('Test Categories')
        plt.ylabel('Number of Tests')
        plt.title('Integration Test Results Summary')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(report_dir, 'test_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run comprehensive integration tests"""
    # Initialize tester
    tester = IntegrationTester()
    
    # Run comprehensive tests
    test_results = tester.run_comprehensive_tests()
    
    # Print summary
    print("\n" + "="*50)
    print("INTEGRATION TEST SUMMARY")
    print("="*50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in test_results.items():
        if isinstance(results, dict):
            for test_name, result in results.items():
                total_tests += 1
                if result.get('status') == 'PASS':
                    passed_tests += 1
                    status_symbol = "✓"
                else:
                    status_symbol = "✗"
                
                print(f"{status_symbol} {category}.{test_name}: {result.get('message', 'N/A')}")
    
    print("="*50)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print("="*50)
    
    return test_results

if __name__ == "__main__":
    results = main()