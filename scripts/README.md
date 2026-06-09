# Scripts - Quản lý AI Models

Thư mục này chứa các scripts để quản lý AI models trong database.

## 📋 Danh sách Scripts

### 1. `sync_models_to_db.py` - Đồng bộ Models vào Database

Script này tự động thêm hoặc cập nhật AI models vào database.

#### Sử dụng:

```bash
# Đồng bộ tất cả models vào database
python scripts/sync_models_to_db.py --sync

# Hiển thị danh sách models hiện có trong database
python scripts/sync_models_to_db.py --list

# Cả hai (sync rồi list)
python scripts/sync_models_to_db.py --sync --list
```

#### Models được đồng bộ:

1. **Helmet Detection** - Phát hiện vi phạm mũ bảo hiểm và áo phản quang
2. **Oil Spill Detection** - Phát hiện tràn dầu
3. **Smoke and Fire Detection** - Phát hiện khói và lửa
4. **General Object Detection** - Phát hiện đối tượng chung (YOLOv8)

#### Tính năng:

- ✅ Tự động tạo bảng database nếu chưa có
- ✅ Cập nhật model nếu đã tồn tại (theo tên + version)
- ✅ Tạo model mới nếu chưa tồn tại
- ✅ Cấu hình sẵn parameters tối ưu:
  - `conf_threshold`: 0.7
  - `iou_threshold`: 0.3
  - Object tracking enabled (30 phút reset)

---

### 2. `update_model_parameters.py` - Cập nhật Parameters

Script này cho phép cập nhật parameters của models đã có trong database.

#### Sử dụng:

```bash
# Cập nhật model cụ thể theo ID
python scripts/update_model_parameters.py \
  --model-id "550e8400-e29b-41d4-a716-446655440000" \
  --conf 0.7 \
  --iou 0.3

# Cập nhật tất cả models cùng loại
python scripts/update_model_parameters.py \
  --model-type helmet_detection \
  --conf 0.7

# Cập nhật tất cả models active
python scripts/update_model_parameters.py \
  --all \
  --conf 0.7 \
  --iou 0.3
```

#### Tham số:

**Selection Options:**
- `--model-id <UUID>` - ID của model cần cập nhật
- `--model-type <type>` - Loại model (helmet_detection, oil_spill, smoke_fire, smoking_behavior, yolo)
- `--all` - Cập nhật tất cả models đang active

**Parameter Options:**
- `--conf <value>` - Confidence threshold (0.0 - 1.0)
- `--iou <value>` - IOU threshold (0.0 - 1.0)

---

## 🚀 Quy trình thiết lập ban đầu

### Bước 1: Đồng bộ models vào database

```bash
# Chạy script đồng bộ
python scripts/sync_models_to_db.py --sync

# Kiểm tra kết quả
python scripts/sync_models_to_db.py --list
```

### Bước 2: (Tùy chọn) Điều chỉnh parameters nếu cần

```bash
# Ví dụ: Giảm confidence threshold cho helmet detection
python scripts/update_model_parameters.py \
  --model-type helmet_detection \
  --conf 0.6
```

### Bước 3: Gán models vào cameras

Sau khi sync xong, bạn có thể gán models vào cameras thông qua API hoặc database.

---

## 📝 Cấu trúc Model trong Database

Mỗi model trong database có các trường sau:

```python
{
  "id": "UUID",
  "name": "Helmet Detection",
  "description": "Detects helmet and vest violations",
  "version": "1.0.0",
  "model_path": "src/ai_models/model_weights/atld_92.pt",
  "parameters": {
    "model_type": "helmet_detection",
    "conf_threshold": 0.7,
    "iou_threshold": 0.3,
    ...
  },
  "is_active": true,
  "is_latest_used": false,
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

---

## 🔧 Thêm Model Mới

Để thêm model mới vào hệ thống:

1. Mở file `scripts/sync_models_to_db.py`
2. Thêm cấu hình model mới vào `MODELS_CONFIG`:

```python
{
    "name": "My New Model",
    "description": "Description of my model",
    "version": "1.0.0",
    "model_path": "src/ai_models/model_weights/my_model.pt",
    "parameters": {
        "model_type": "my_model_type",
        "conf_threshold": 0.7,
        "iou_threshold": 0.3,
        # Add other parameters...
    },
    "is_active": True,
}
```

3. Chạy lại script sync:

```bash
python scripts/sync_models_to_db.py --sync
```

---

## 📊 Model Types Hiện có

| Model Type | Description | Tracking |
|------------|-------------|----------|
| `helmet_detection` | Helmet and vest violations | ✅ Yes (30min) |
| `oil_spill` | Oil spill detection | ✅ Yes (30min) |
| `smoke_fire` | Smoke and fire detection | ✅ Yes (30min) |
| `smoking_behavior` | Smoking behavior detection | ⚠️ OpenVINO ONNX pair |
| `yolo` | General object detection | ✅ Yes (30min) |

---

## ⚠️ Lưu ý

1. **Backup Database**: Nên backup database trước khi chạy scripts lần đầu
2. **Model Weights**: Đảm bảo các file model weights đã được đặt đúng vị trí (`.pt` hoặc `.onnx` theo từng model)
3. **Database Connection**: Kiểm tra kết nối database trong `.env` hoặc config
4. **Permissions**: Script cần quyền ghi vào database

---

## 🐛 Troubleshooting

### Lỗi: "No module named 'src'"

```bash
# Đảm bảo chạy từ thư mục root của project
cd /home/dat/2025/petrolimex-ai-backend
python scripts/sync_models_to_db.py
```

### Lỗi: "Could not connect to database"

Kiểm tra cấu hình database trong file `.env` hoặc `src/database.py`

### Model không được cập nhật

Script chỉ cập nhật nếu `name` và `version` trùng khớp. Để force update, thay đổi version number.

---

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Log output từ script
2. Database connection
3. Model weights file paths
4. Python dependencies (sqlalchemy, etc.)
