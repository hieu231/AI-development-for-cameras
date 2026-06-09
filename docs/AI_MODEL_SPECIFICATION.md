# Tài liệu đặc tả các model AI

## 1. Mục tiêu
Tài liệu này đặc tả hành vi runtime của 4 model AI trong backend:
- TranDauModel
- SmokeFireModel
- PetrolimexDetectionModel
- PeopleControlModel

Phạm vi đặc tả tập trung vào:
- Đầu vào, tham số cấu hình và đầu ra chuẩn DetectionResult
- Logic lọc nhiễu, theo dõi đối tượng, cơ chế dedup sự kiện
- Quy tắc phát sinh cảnh báo và metadata gửi xuống pipeline sự kiện

Không bao gồm:
- Hướng dẫn train model
- Đặc tả CSDL và API bên ngoài backend AI

## 2. Kiến trúc chung
Tất cả model đề cập trong tài liệu này đều kế thừa BaseModel và trả về DetectionResult theo mẫu:
- frame: frame đã vẽ annotation nếu annotate=True, ngược lại là None
- event: true nếu có sự kiện mới cần ghi nhận trong frame hiện tại
- metadata: dict mô tả sự kiện, danh sách detections, violations, thông tin tổng hợp

Các thành phần runtime dùng chung:
- Ultralytics YOLO cho detect và track
- ROI gate: chỉ xử lý detection có tâm nằm trong ROI hợp lệ
- ObjectTracker và/hoặc RecentViolationDeduplicator để tránh lặp sự kiện
- Khóa luồng truy cập state bằng lock để đảm bảo thread-safe

## 3. Quy ước đầu vào và đầu ra
### 3.1 Đầu vào process_frame
- frame: numpy ndarray (BGR)
- kwargs thông dụng:
  - annotate: bool, mặc định True
  - roi: 1 ROI hoặc tập ROI, được quy đổi về tọa độ pixel
  - inference_frame: (chỉ áp dụng ở SmokeFireModel) frame sạch cho suy luận

### 3.2 Quy ước ROI
- Nếu detection center không nằm trong ROI, detection bị loại
- ROI có thể là polygon đơn hoặc danh sách polygon
- Nếu ROI không hợp lệ thì model có thể fallback toàn khung hình (tùy model)

### 3.3 Đầu ra metadata
Các model có format metadata riêng, nhưng đều có tính chất chung:
- Có danh sách detections
- Có danh sách violations (các sự kiện mới được xác nhận)
- Có type/eventType/title/description để phục vụ hiển thị/cảnh báo

## 4. TranDauModel
### 4.1 Mục đích
Phát hiện dòng chảy tràn dầu đột ngột, ưu tiên cảnh báo nhanh và xử lý tức thời.

### 4.2 Model và thành phần phụ
- Model chính: tran_dau.pt
- Model phụ foreground: yolo11n.pt (người/xe) để lọc false positive

### 4.3 Tham số quan trọng
- conf_threshold, iou_threshold
- detection_cooldown
- foreground_conf_threshold, foreground_iou_threshold
- foreground_cover_ratio_threshold, foreground_proximity_ratio
- recent_foreground_hold_seconds
- global_event_cooldown, continuous_event_interval
- event_dedup_window_seconds, event_dedup_iou_threshold
- fallback_iou_threshold (cho key fallback khi track_id None)

### 4.4 Luồng xử lý frame
1. Tạo gray frame (có Gaussian blur nếu cấu hình)
2. Build ROI polygons
3. Track với model chính
4. Mỗi bbox:
   - Validate kích thước bbox
   - Kiểm tra tâm bbox trong ROI
   - Resolve detection key ổn định:
     - Nếu có track_id: dùng trực tiếp
     - Nếu không có track_id: match IoU với pending detection trong negative namespace
5. Tạo detection object với severity=high, response_required=True
6. Kích hoạt gate sự kiện:
   - Loại bỏ duplicate ngắn hạn qua RecentViolationDeduplicator
   - Kiểm tra ObjectTracker theo track
   - Kiểm tra global gate theo global_event_cooldown và continuous_event_interval
7. Nếu qua gate: tạo violation SUDDEN_OIL_FLOW_DETECTED
8. Vẽ bbox và ROI nếu annotate=True
9. Build metadata và trả DetectionResult

### 4.5 Cơ chế dedup
- Dedup theo cửa sổ thời gian ngắn và IoU để tránh event trùng theo frame
- Có gate theo track và gate toàn cục để tránh spam
- Fallback ID dùng không gian số âm, tránh trùng với track ID thật

### 4.6 Metadata đặc trưng
- incident_type: sudden_oil_flow
- response_mode: immediate
- requires_immediate_response: true khi có detection
- alert.level: HIGH
- violations có event_kind: SUDDEN_OIL_FLOW_DETECTED

## 5. SmokeFireModel
### 5.1 Mục đích
Phát hiện khói và lửa trong ROI, tối ưu recall cho khói nhưng giảm false positive bằng nhiều lớp lọc.

### 5.2 Model và thành phần phụ
- Model chính: YOLOv10-FireSmoke-X.pt
- Model phụ foreground: yolo11n.pt

### 5.3 Tham số quan trọng
- fire_conf_threshold, smoke_conf_threshold, track_conf_threshold
- smoke_persist_seconds, smoke_stale_timeout
- smoke_max_saturation
- min_bbox_area_ratio, max_bbox_area_ratio
- tail_light_* (bộ lọc đèn hậu xe cho class fire)
- foreground_* và recent_foreground_hold_seconds
- detection_cooldown, save_cooldown
- event_dedup_window_seconds, event_dedup_iou_threshold

### 5.4 Luồng xử lý frame
1. Chuyển ROI normalized sang polygon pixel
2. Lấy clean_frame (ưu tiên inference_frame nếu có)
3. Phát hiện foreground (người/xe) và lưu recent foreground boxes
4. Track khói/lửa với model chính
5. Mỗi detection hợp lệ class và conf:
   - Kiểm tra ROI
   - Lọc area ratio quá nhỏ/quá lớn
   - Lọc tail-light false positive (chỉ fire nhỏ)
   - Lọc colorful false positive cho smoke (HSV saturation)
   - Lọc overlap/recent foreground
6. Resolve track_id fallback bằng IoU nếu tracker bị fail
7. Temporal gate:
   - Fire: xác nhận ngay
   - Smoke: phải duy trì liên tục >= smoke_persist_seconds
8. Dedup event:
   - Dedup ngắn hạn theo RecentViolationDeduplicator
   - Dedup theo ObjectTracker (theo track + class)
9. Tạo violation với event_type và description phù hợp fire/smoke
10. Vẽ bbox và ROI, cleanup stale state

### 5.5 Cơ chế giảm false positive
- Lọc theo màu cho smoke
- Lọc đèn hậu xe cho fire
- Lọc overlap với người/xe
- Temporal persistence cho smoke

### 5.6 Metadata đặc trưng
- type/eventType/title = event_type của sự kiện chính
- detections: class_name tiếng Việt (Lửa/Khói)
- violations: có event_type và description chi tiết
- model_type: smoke_fire

## 6. PetrolimexDetectionModel
### 6.1 Mục đích
Phát hiện vi phạm PPE trong khu vực giám sát:
- Không mặc áo bảo hộ
- Không đội mũ bảo hộ

### 6.2 Model và thành phần phụ
- Model chính PPE vest/no-vest: petrolimex.pt
- Model no_hardhat: atld_92.pt
- Model people aux (tùy chọn): yolo11m.pt

### 6.3 Tham số quan trọng
- conf_threshold, novest_conf_threshold
- no_hardhat_conf_threshold
- enable_people_aux_detection, people_conf_threshold
- people_match_iou_threshold
- head_region_fraction (ràng buộc vị trí no_hardhat trên đầu người)
- detection_cooldown, save_cooldown
- track_match_iou_threshold, track_stale_timeout
- event_dedup_window_seconds, event_dedup_iou_threshold

### 6.4 Luồng xử lý frame
1. Track PPE vest/no-vest trên model chính
2. Normalize class labels qua bảng alias (vest, novest, no_hardhat...)
3. Lọc theo conf và ROI
4. Ổn định track_id theo namespace riêng cho vest/hardhat/people
5. Nếu bật people aux:
   - Track person
   - Match person với detections vest/novest bằng score
   - Tạo people_status positive/negative
   - Ẩn bớt novest detection khi bị people aux xử lý
6. Chạy model no_hardhat:
   - Lọc ROI
   - Nếu có person boxes thì bắt buộc detection no_hardhat nằm trong head region
7. Tổng hợp violations và dedup:
   - Dedup ngắn hạn bằng bộ nhớ recent emitted
   - Gate theo ObjectTracker
8. Tạo events_to_create và metadata

### 6.5 Cơ chế dedup
- _recent_emitted_violations dedup theo thời gian + IoU + track
- ObjectTracker gate để tránh lặp theo cooldown
- Stable tracks tách riêng từng namespace model

### 6.6 Metadata đặc trưng
- type/eventType/title: Không tuân thủ bảo hộ lao động
- detections: tất cả detection PPE
- people_statuses: kết quả person positive/negative
- violations: danh sách event đã được ghi nhận
- model_type: petrolimex_detection_model

## 7. PeopleControlModel
### 7.1 Mục đích
Phát hiện người xâm nhập trái phép vào khu vực cấm, hỗ trợ nhiều ROI.

### 7.2 Model và tham số quan trọng
- Model: yolo11m.pt (class person)
- conf_threshold
- detection_cooldown
- global_event_cooldown (hiện lưu state, logic sự kiện ưu tiên per-track)
- track_lost_timeout_sec
- max_roi_polygons (1 đến 5)
- event_dedup_window_seconds, event_dedup_iou_threshold

### 7.3 Luồng xử lý frame
1. Build danh sách ROI polygons từ normalized ROI
2. Vẽ ROI nếu annotate=True
3. Track person class trên clean frame
4. Mỗi person bbox:
   - Tìm roi_index có chứa center
   - Khởi tạo/cập nhật track_status
   - Nếu vào ROI lần đầu hoặc qua cooldown track: có thể trigger
   - Dedup ngắn hạn bằng RecentViolationDeduplicator
   - Nếu trigger: tạo detection_result sự kiện
5. Cleanup track bị mất quá track_lost_timeout_sec
6. Build all_detections trong ROI
7. Build violations từ sự kiện vừa trigger
8. Build metadata và trả kết quả

### 7.4 Đặc điểm logic sự kiện
- Ưu tiên cooldown theo từng track (per-track)
- Không khóa toàn cục giữa nhiều đối tượng độc lập
- Hỗ trợ đa ROI, gán roi_index theo vị trí center

### 7.5 Metadata đặc trưng
- event type: Xâm nhập trái phép khu vực
- detections: danh sách person trong ROI
- violations: có roi_index, bbox, confidence
- roi_count và max_roi_polygons

## 8. Ràng buộc chất lượng và vận hành
### 8.1 Thread safety
- TranDauModel: dùng RLock cho state mutable
- SmokeFireModel, PetrolimexDetectionModel, PeopleControlModel: dùng Lock

### 8.2 Khuyến nghị cấu hình production
- Chọn conf threshold theo từng camera thay vì dùng giá trị toàn hệ thống
- Bật event dedup window > 0 để giảm spam sự kiện
- Đặt detection_cooldown phù hợp theo nghiệp vụ:
  - Các sự kiện khẩn cấp (tràn dầu, cháy): cooldown ngắn
  - Các sự kiện PPE và intrusion: cooldown vừa đủ để tránh trùng lặp

### 8.3 Kiểm thử để xác nhận hành vi
- Kiểm thử unit cho logic dedup và resolve track ID fallback
- Kiểm thử integration với stream thực tế có ROI động
- Kiểm thử metadata contract để đảm bảo tương thích API lưu sự kiện

## 9. Contract metadata tối thiểu để tích hợp
Mốc tối thiểu khuyến nghị cho hệ thống tích hợp xuống dưới:
- event bool
- metadata.eventType
- metadata.description
- metadata.timestamp
- metadata.detections[]
- metadata.violations[]
- metadata.model_type

## 10. Ghi chú triển khai
Tài liệu này đặc tả hành vi theo code hiện tại trong các file:
- src/ai_models/tran_dau_model.py
- src/ai_models/smoke_fire_model.py
- src/ai_models/petrolimex_detection_model.py
- src/ai_models/people_control_model.py

Khi thay đổi logic model, cần cập nhật lại tài liệu này để đồng bộ với contract metadata và logic event.