# VSS Debugging Guide

## 1. `play_url` bi `null`

Nguyen nhan thuong gap:

- Camera chua duoc tao trong DB.
- Camera da tao nhung `vss_device_id` khong trung `device_id` request.
- Camera da tao nhung `vss_channel` khong trung `channel` request.

Checklist:

1. Kiem tra `device_id` va `channel` trong request `POST /api/vss/build-stream-url`.
2. Kiem tra ban ghi camera trong bang `cameras` co `vss_device_id` va `vss_channel` trung khop.
3. Goi lai endpoint sau khi da map camera.

Ky vong:

- `offer_url` van co gia tri.
- `play_url` la `null` cho den khi mapping dung.

## 2. `offer_url` sai host

Nguyen nhan:

- Reverse proxy / base URL forwarding cau hinh chua dung.

Checklist:

1. Kiem tra `request.base_url` ma FastAPI nhan duoc.
2. Neu dung Nginx/Traefik, dam bao forward dung `Host`, `X-Forwarded-Proto`, `X-Forwarded-For`.
3. So sanh `offer_url` tra ve voi domain ma FE dang goi API.

Ky vong:

- `offer_url` phai cung origin voi API response.

## 3. Co `play_url` nhung mo khong len hinh

Checklist:

1. Camera da start chua.
2. `GET /api/stream/camera/{camera_id}/status` co `online=true` khong.
3. `GET /api/webrtc/status` co `enabled=true` va `aiortc_available=true` khong.
4. Mo truc tiep `play_url` tren browser cung origin.
5. Neu FE tu dam nhan WebRTC, gui SDP offer vao `offer_url` va kiem tra response answer.

Ky vong:

- WebRTC player mo duoc trang.
- Browser nhan video track.
- Camera dang chay thi khong duoc 404 o `POST /api/webrtc/offer`.

## 4. `stream_url` hop le nhung camera khong doc duoc

Checklist:

1. Token trong `stream_url` con han khong.
2. URL con day du `deviceId`, `chs`, `stream`, `wnum`, `panel`, `buffer` khong.
3. VSS co dang rate limit login khong.
4. Thu `POST /api/vss/resolve` neu can build FLV URL cho pipeline FFmpeg.

Ky vong:

- `stream_url` la RealVideo URL co token hop le.
- Loi VSS rate limit phai duoc map ve HTTP `429`.

## 5. Tieu chi kiem soat khi release

- Co test hoi quy cho `offer_url` va `play_url`.
- Swagger docs hien dung vi du response moi.
- FE da xac nhan doc duoc `play_url = null` ma khong bi vo flow.
- FE da xac nhan uu tien `play_url` neu co.
- QA da kiem tra 1 case co mapping camera va 1 case khong co mapping.

## 6. Tieu chi thanh cong khi nghiem thu

- Case 1: Camera da map
	- `build-stream-url` tra du `stream_url`, `offer_url`, `play_url`.
	- Mo `play_url` xem duoc hinh annotate.

- Case 2: Camera chua map
	- `build-stream-url` tra `stream_url`, `offer_url`, `play_url = null`.
	- FE hien thong bao dung va khong crash.

- Case 3: VSS rate limit
	- Endpoint tra `429`.

- Case 4: Request sai format
	- Endpoint tra `422`.
