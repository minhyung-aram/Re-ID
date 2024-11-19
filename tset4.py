import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from collections import deque
import time

class FastReIDInterface:
    def __init__(self, config_file, weights_path, device, batch_size=32):
        self.device = device
        self.batch_size = batch_size
        
        # FastReID 설정 및 모델 초기화
        self.cfg = self._setup_cfg(config_file, weights_path)
        self.model = build_model(self.cfg)
        self.model.eval()
        Checkpointer(self.model).load(weights_path)
        
        if self.device == 'cuda':
            self.model = self.model.eval().cuda().half()  # 모델을 half precision으로 변환
        else:
            self.model = self.model.eval()
        
        self.size_test = self.cfg.INPUT.SIZE_TEST
        
        # 정규화 값을 텐서로 미리 변환하여 GPU에 저장
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        
        if self.device == 'cuda':
            self.pixel_mean = self.pixel_mean.half()
            self.pixel_std = self.pixel_std.half()

    def _setup_cfg(self, config_file, weights_path):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS = weights_path
        cfg.freeze()
        return cfg
    
    @torch.no_grad()
    def process_images_batch(self, image, detections, min_box_area=100, conf_thres=0.5):
        """배치 단위로 이미지 처리 (타입 일치 수정)"""
        if detections is None or len(detections) == 0:
            return np.array([]), []

        H, W, _ = image.shape
        batch_imgs = []
        valid_dets = []
        
        for det in detections:
            if det[4] < conf_thres:  # confidence threshold 적용
                continue
                
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            # 작은 박스 필터링
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_box_area or x2 <= x1 or y2 <= y1:
                continue

            im = image[y1:y2, x1:x2]
            
            # FastReID 입력 크기에 맞게 리사이즈
            im = cv2.resize(im, (self.cfg.INPUT.SIZE_TEST[1], self.cfg.INPUT.SIZE_TEST[0]), 
                          interpolation=cv2.INTER_LINEAR)
            im = im[:, :, ::-1]  # BGR to RGB
            im = im.transpose(2, 0, 1)  # HWC to CHW
            
            batch_imgs.append(im)
            valid_dets.append(det)

        if not batch_imgs:
            return np.array([]), []

        # numpy array를 torch tensor로 변환
        batch_imgs = torch.from_numpy(np.stack(batch_imgs))
        
        # GPU로 전송 및 데이터 타입 변환
        if self.device == 'cuda':
            batch_imgs = batch_imgs.cuda().half()  
        else:
            batch_imgs = batch_imgs.float()
        
        batch_imgs = (batch_imgs - self.pixel_mean) / self.pixel_std

        # 배치 단위 특징 추출
        features = []
        for i in range(0, len(batch_imgs), self.batch_size):
            batch = batch_imgs[i:i + self.batch_size]
            feat = self.model(batch)
            feat = F.normalize(feat, p=2, dim=1)
            features.append(feat)

        if features:
            features = torch.cat(features, dim=0).cpu().numpy()
            return features, valid_dets

        return np.array([]), []

class Track:
    """트랙 객체 관리를 위한 클래스"""
    def __init__(self, feature, bbox, track_id):
        self.id = track_id
        self.bbox = bbox
        self.feature_history = deque(maxlen=10)  # 특징 벡터 이력 관리
        self.feature_history.append(feature)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = 'tentative'  # tentative, confirmed, deleted
        
    def update(self, feature, bbox):
        self.feature_history.append(feature)
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
            
    def get_feature(self):
        return np.mean(self.feature_history, axis=0)

class CameraTracker:
    """단일 카메라 트래커"""
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.tracks = {}
        self.next_id = 0
        self.max_age = 30
        self.min_hits = 3
        self._color_cache = {}

    def get_track_id(self, global_id):
        """카메라별 고유 ID 생성"""
        return f"Cam{self.camera_id}-{global_id}"

    def _get_color(self, idx):
        if idx not in self._color_cache:
            np.random.seed(int(idx) * 777)
            self._color_cache[idx] = tuple(map(int, np.random.randint(100, 255, 3)))
        return self._color_cache[idx]

class MultiCameraTracker:
    def __init__(self, config_file, weights_path, device='cuda'):
        self.device = device
        self.reid = FastReIDInterface(config_file, weights_path, device)
        self.detector = YOLO("yolov8s.pt")
        self.detector.to(device)
        
        # 트래킹 파라미터
        self.cos_threshold = 0.6
        self.conf_thres = 0.5
        
        # 각 카메라별 트래커 저장
        self.camera_trackers = {}
        
        # 디스플레이 설정
        self.display_width = 1100
        self.display_height = 600

    @torch.no_grad()
    def process_frame(self, frame, camera_id):
        """단일 프레임 처리"""
        results = self.detector(frame, classes=[0], conf=self.conf_thres)[0]
        detections = results.boxes.data.cpu().numpy()
        
        if len(detections) == 0:
            return np.array([]), [], []
            
        features, valid_dets = self.reid.process_images_batch(
            frame, detections,
            min_box_area=100,
            conf_thres=self.conf_thres
        )
        
        return features, valid_dets, [det[4] for det in valid_dets]

    def update_tracks(self, camera_id, features, dets, confs):
        """카메라별 트랙 업데이트"""
        # 카메라 트래커가 없으면 새로 생성
        if camera_id not in self.camera_trackers:
            self.camera_trackers[camera_id] = CameraTracker(camera_id)
        
        tracker = self.camera_trackers[camera_id]
        
        if not features.size or not tracker.tracks:
            # 새로운 트랙 생성
            for feat, det, conf in zip(features, dets, confs):
                if conf > self.conf_thres:
                    track_id = tracker.next_id
                    tracker.tracks[track_id] = Track(feat, det, track_id)
                    tracker.next_id += 1
            return

        # 현재 활성 트랙 특징 벡터 추출
        track_features = []
        track_ids = []
        
        for track_id, track in list(tracker.tracks.items()):
            if track.state != 'deleted':
                track_features.append(track.get_feature())
                track_ids.append(track_id)

        if not track_features:
            return

        track_features = np.array(track_features)
        
        # 코사인 유사도 계산
        cost_matrix = 1 - np.dot(features, track_features.T)
        cost_matrix[cost_matrix > (1 - self.cos_threshold)] = 1e5

        # 헝가리안 알고리즘으로 매칭
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1 - self.cos_threshold):
                track_id = track_ids[c]
                tracker.tracks[track_id].update(features[r], dets[r])
                matched.add(r)

        # 미매칭 detection에 대한 새로운 트랙 생성
        for i, (feat, det, conf) in enumerate(zip(features, dets, confs)):
            if i not in matched and conf > self.conf_thres:
                track_id = tracker.next_id
                tracker.tracks[track_id] = Track(feat, det, track_id)
                tracker.next_id += 1

        # 트랙 상태 업데이트 및 삭제된 트랙 제거
        tracker.tracks = {k: v for k, v in tracker.tracks.items() 
                        if not (v.time_since_update > tracker.max_age 
                               or v.state == 'deleted')}
        
        for track in tracker.tracks.values():
            track.time_since_update += 1

    def visualize_results(self, frames):
        """결과 시각화"""
        num_cameras = len(frames)
        frame_height = self.display_height
        frame_width = self.display_width // num_cameras

        vis_frames = []
        for camera_id, frame in enumerate(frames):
            frame = frame.copy()
            tracker = self.camera_trackers.get(camera_id)
            
            if tracker:
                # 확인된 트랙만 표시
                for track_id, track in tracker.tracks.items():
                    if track.state == 'confirmed':
                        x1, y1, x2, y2 = map(int, track.bbox[:4])
                        color = tracker._get_color(track_id)
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # ID 텍스트
                        text = f"ID:{tracker.get_track_id(track_id)}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - text_size[1] - 8), 
                                    (x1 + text_size[0], y1), color, -1)
                        cv2.putText(frame, text, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 프레임 크기 조정
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            # 카메라 레이블
            cv2.putText(frame, f"Camera {camera_id+1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            vis_frames.append(frame)

        return np.hstack(vis_frames)

    def run(self, video_paths):
        """다중 카메라 트래킹 실행"""
        caps = [cv2.VideoCapture(p) for p in video_paths]
        if not all(cap.isOpened() for cap in caps):
            print("Error: Could not open all video files")
            return

        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                rets_frames = [cap.read() for cap in caps]
                if not all(ret for ret, frame in rets_frames):
                    break
                    
                frames = [frame for ret, frame in rets_frames]
                
                # 각 카메라 프레임 처리
                for camera_id, frame in enumerate(frames):
                    features, dets, confs = self.process_frame(frame, camera_id)
                    if len(features) > 0:
                        self.update_tracks(camera_id, features, dets, confs)

                # 결과 시각화
                result_frame = self.visualize_results(frames)
                
                # FPS 계산 및 표시
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    cv2.putText(result_frame, f'FPS: {fps:.1f}', 
                              (10, self.display_height - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 결과 표시
                cv2.imshow('Multi-Camera Tracking', result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            elapsed_time = time.time() - start_time
            print(f"\nProcessing completed:")
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {frame_count / elapsed_time:.1f}")
            
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()

def main():
    config_file = "D:/mc-mot/fast-reid/configs/Base-bagtricks.yml"
    weights_path = "D:/mc-mot/fast-reid/pretrained/market_aic_bot_R50.pth"

    tracker = MultiCameraTracker(
        config_file=config_file,
        weights_path=weights_path,
        device='cuda'
    )

    video_paths = [
        "../dataset/cam1.avi",
        "../dataset/cam2.avi",
        "../dataset/cam3.avi"
    ]
    
    print("Starting multi-camera tracking...")
    tracker.run(video_paths)

if __name__ == "__main__":
    main()