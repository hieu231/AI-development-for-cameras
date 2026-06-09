# VSS Quick Start

## Muc tieu

Tai lieu nay chot contract cho luong VSS -> AI Backend -> Frontend sau thay doi `POST /api/vss/build-stream-url`.

Frontend khong duoc xem `stream_url` la link playback sau cung. `stream_url` la link nguon VSS da co token. Bo link playback cho WebRTC duoc tra qua `offer_url` va `play_url`.

## 1. Request build stream

Endpoint:

```http
POST /api/vss/build-stream-url
Content-Type: application/json
```

Request body:

```json
{
	"base_url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
	"username": "TEST1",
	"password": "4de93544234adffbb681ed60ffcfb941",
	"channel": "1",
	"device_id": "HAN3-20-7203"
}
```

## 2. Response stream bundle

Response body:

```json
{
	"id": "550e8400-e29b-41d4-a716-446655440000",
	"stream_url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html?token=abc123&deviceId=HAN3-20-7203&chs=1&stream=1&wnum=1&panel=0&buffer=2000",
	"token": "abc123",
	"device_id": "HAN3-20-7203",
	"channel": "1",
	"offer_url": "http://localhost:8668/api/webrtc/offer",
	"play_url": "http://localhost:8668/api/webrtc/play/2f4f7ec1-6f46-4f53-9a95-f3b0ff8e7f8d"
}
```

## 3. Y nghia tung truong

- `id`: request id de trace log va correlate FE/BE.
- `stream_url`: RealVideo URL da duoc BE cap token. Dung cho backend/camera source, khong dung lam player URL cho trinh duyet.
- `token`: token VSS sau login.
- `device_id`: VSS device id.
- `channel`: kenh VSS.
- `offer_url`: endpoint co dinh de FE gui WebRTC SDP offer.
- `play_url`: trang player WebRTC san co. Chi co gia tri khi he thong map duoc `device_id + channel` sang camera trong DB.

## 4. Dinh dang luong camera duoc chap nhan

Frontend phai chap nhan 2 cap su dung sau:

1. Cap nguon VSS
	 - `stream_url`
	 - `token`

2. Cap playback AI stream
	 - `offer_url`
	 - `play_url`

Quy tac:

- Neu `play_url` khac `null`, FE uu tien dung WebRTC.
- Neu `play_url` la `null`, FE khong duoc gia dinh camera play duoc ngay. Can thong bao camera chua duoc map trong he thong hoac fallback sang endpoint camera links sau khi camera duoc tao.
- `offer_url` co the co gia tri ngay ca khi `play_url = null` vi no la endpoint chung cua he thong.
- FE khong duoc tu ghep `play_url` tu `device_id`; chi duoc dung `play_url` BE tra ve.

## 5. Luong FE de nghi

1. Goi `POST /api/vss/build-stream-url`.
2. Luu `stream_url` vao cau hinh camera neu dang tao camera moi.
3. Neu response co `play_url`, mo player hoac bat dau WebRTC negotiation qua `offer_url`.
4. Neu `play_url = null`, tao hoac cap nhat camera trong he thong de tao mapping `vss_device_id + vss_channel` -> `camera_id`, sau do lay lai stream bundle.

## 6. Tieu chi kiem soat

- `base_url` phai parse duoc va co scheme + hostname hop le.
- `stream_url` phai chua day du: `token`, `deviceId`, `chs`, `stream`, `wnum`, `panel`, `buffer`.
- `offer_url` phai cung origin voi API dang tra response.
- `play_url` chi duoc tra khi ton tai camera trong DB co `vss_device_id == device_id` va `vss_channel == channel`.
- `play_url` phai theo dung format `/api/webrtc/play/{camera_id}`.
- Khi khong map duoc camera, `play_url` phai la `null`, khong duoc tra link gia.
- Truong hop VSS rate limit phai tra HTTP `429`.
- Truong hop loi parse/validation phai tra HTTP `422`.
- Truong hop loi runtime khac phai tra HTTP `500` hoac `503` theo ngu canh.

## 7. Tieu chi thanh cong

- Swagger hien dung example response co `offer_url` va `play_url`.
- FE nhan duoc du 3 loai link trong mot lan goi: `stream_url`, `offer_url`, `play_url`.
- Neu camera da duoc map, mo [src/api/webrtc.py](src/api/webrtc.py) endpoint `play/{camera_id}` xem duoc video annotate.
- `POST /api/webrtc/offer` thanh cong voi camera dang chay va tra SDP answer hop le.
- Test hoi quy cho `tests/test_vss_api.py` pass.
- Khong pha vo backward compatibility cua code goi truc tiep `build_stream_url(...)` trong test/noi bo.

## 8. Ghi chu tich hop

- `play_url` la URL de mo player HTML WebRTC co san.
- `offer_url` la URL de FE tu tuyen chinh WebRTC neu dung player rieng.
- Neu FE can danh sach link cho camera da ton tai, dung them `GET /api/stream/camera/{camera_id}/links`.
