import os
import cv2
import numpy as np
import torch


class FaceExtractor:
    def __init__(self, face_size=224, device=None):
        # Set device if not provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.use_cuda = (device.type == 'cuda')
        MODEL_DIR = r"C:\Users\rajmo\deepfake"
        
        # Load face detection model (DNN based for stability)
        prototxt_path = os.path.join(MODEL_DIR, 'deploy.prototxt')
        caffemodel_path = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # Download models if not present
        if not os.path.exists(prototxt_path):
            os.system("wget -O " + prototxt_path + r"deploy.prototxt")
        
        
        if not os.path.exists(caffemodel_path):
            os.system("wget -O " + caffemodel_path + r"res10_300x300_ssd_iter_140000.caffemodel")
        
        self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        
        # Set OpenCV DNN backend to CUDA if available
        if self.use_cuda:
            # Check if OpenCV is built with CUDA support
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA for face detection")
            else:
                print("OpenCV not built with CUDA support, using CPU instead")
        
        self.face_size = face_size
    
    def extract_face(self, frame, confidence_threshold=0.5):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        # Find the highest confidence face
        best_face = None
        best_confidence = confidence_threshold
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_confidence:
                best_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Expand the box by 20%
                width, height = x2 - x1, y2 - y1
                x1 = max(0, x1 - int(width * 0.1))
                y1 = max(0, y1 - int(height * 0.1))
                x2 = min(w, x2 + int(width * 0.1))
                y2 = min(h, y2 + int(height * 0.1))
                
                if x2 - x1 > 0 and y2 - y1 > 0:
                    face = frame[y1:y2, x1:x2]
                    try:
                        face = cv2.resize(face, (self.face_size, self.face_size))
                        best_face = face
                        best_box = (x1, y1, x2, y2)
                    except Exception as e:
                        continue
        
        return best_face, best_box

    def extract_faces_from_video(self, video_path, num_frames=16, device=None):
        # Update device if provided
        if device is not None:
            self.device = device
            self.use_cuda = (device.type == 'cuda')
            if self.use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return None
        
        # Extract frames at regular intervals
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        faces = []
        
        # Process in batches for better GPU utilization
        batch_size = 16 if self.use_cuda else 1  # Larger batch size for GPU
        
        for i in range(0, len(frame_indices), batch_size):
            # Process a batch of frames
            batch_indices = frame_indices[i:i+batch_size]
            batch_faces = []
            
            for idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Clean up memory
                if self.use_cuda:
                    torch.cuda.empty_cache()
                    
                face, _ = self.extract_face(frame)
                if face is not None:
                    batch_faces.append(face)
            
            faces.extend(batch_faces)
        
        cap.release()
        # If no faces were found, return None
        if len(faces) == 0:
            return None
            
        # If some faces were detected but not enough, duplicate the last one
        while len(faces) < num_frames and len(faces) > 0:
            faces.append(faces[-1])
            
        # Return only if we have enough faces
        if len(faces) >= num_frames:
            return np.array(faces[:num_frames])
        else:
            return None

    def process_video_batch(self, video_paths, num_frames=16):
        """Process multiple videos in parallel for better GPU utilization"""
        results = {}
        
        for path in video_paths:
            faces = self.extract_faces_from_video(path, num_frames)
            results[path] = faces
            
            # Clean up memory
            if self.use_cuda:
                torch.cuda.empty_cache()
                
        return results
        