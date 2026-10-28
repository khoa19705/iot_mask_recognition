# MaskLens — Webcam Mask Detection

## Lab 3

Đã bổ sung đo thời gian suy luận, phản hồi và toàn trình; trung bình/P95, xuất CSV phiên đo, kiểm tra nhãn model và cấu hình Docker/Cloud.
Xem [hướng dẫn Lab 3](LAB3.md) để chạy, triển khai và lấy số liệu. Mặc định vẫn chạy **local**; chỉ có số đo Cloud thật sau khi triển khai lên máy chủ Cloud.

### Triển khai repo này lên Render

Repository đã chứa `best.onnx` cùng Dockerfile tại gốc. Docker dùng ONNX Runtime, không cài/nạp PyTorch; chạy một luồng CPU và ảnh suy luận 320 px để giảm tải. Chọn **Web Service**, kết nối repository `khoa19705/iot_mask_recognition`, chọn nhánh `codex/lab3-cloud`, dùng runtime **Docker** và để trống **Root Directory**.
Đặt biến môi trường `APP_ENV=cloud`, `FLASK_DEBUG=0`, `MAX_UPLOAD_MB=8`; health check `/api/health`.
Không cần nhập Build Command/Start Command khi dùng Dockerfile, cũng không cần `MODEL_URL` vì trọng số đã có trong repository.
Không đặt `MASK_MODEL_PATH=best.pt` trên Render: Docker mới dùng mặc định `best.onnx`. Nếu đã đặt biến này, sửa thành `best.onnx`. Không có gói trả phí nào được tự đăng ký. Gói 0,1 CPU vẫn có thể xử lý chậm; bản tối ưu chưa đảm bảo realtime hoặc đủ RAM trên mọi máy chủ.

Chỉ bật nhận diện ở một tab. Nếu có yêu cầu nhận diện đồng thời, API trả 429 để không xếp hàng làm nghẽn health check. Ảnh gửi tối đa rộng 640 px, JPEG 70%; ảnh đưa vào model 320 px có thể giảm nhận diện khuôn mặt nhỏ.

Ứng dụng Flask dùng Ultralytics YOLO để nhận diện khẩu trang trực tiếp trên một webcam. Khung bounding box được vẽ đè lên video đang chạy; không tải ảnh/video và không mở camera kết quả thứ hai.

## Chạy ứng dụng

Trong PowerShell tại thư mục project:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Mở [http://localhost:5000](http://localhost:5000), cấp quyền webcam cho trình duyệt và bấm **Bật webcam**. Webcam hoạt động trên `localhost` hoặc trang HTTPS. Trọng số YOLO mặc định là `best.pt` ở thư mục project; nếu đặt model ở nơi khác, chỉnh `MASK_MODEL_PATH` trong `.env` rồi khởi động lại server.

## Cách hoạt động

1. Trình duyệt đọc webcam và gửi một khung hình mỗi lần suy luận hoàn tất.
2. Flask chạy YOLO trên khung hình đó và trả về bounding boxes, lớp và confidence.
3. Trình duyệt vẽ bounding boxes trực tiếp lên lớp canvas trong suốt phủ trên webcam; video gốc vẫn phát liên tục.
4. Thống kê bên cạnh luôn phản ánh khung hình vừa phân tích.

## Log phát hiện

- Khi số lượng theo lớp thay đổi, server ghi một sự kiện vào `logs/detections.jsonl`.
- Nếu trạng thái không đổi, server nhắc lại sự kiện sau mỗi 10 giây; khung hình không có phát hiện không được ghi log.
- Trang hiển thị các sự kiện gần nhất và nút **Tải log** tải toàn bộ JSONL.
- Mỗi sự kiện có thời gian UTC, số lượng theo lớp và detections gồm nhãn, confidence, bounding box.

Các API: `GET /api/health`, `POST /api/detect`, `GET /api/logs`, `GET /api/logs/download`.

## Cấu hình

```dotenv
MASK_MODEL_PATH=best.pt
PORT=5000
FLASK_DEBUG=0
```

Chỉ bật `FLASK_DEBUG=1` trong môi trường phát triển cục bộ. `best.pt` dành cho local/huấn luyện và `best.onnx` dành cho Cloud có sẵn trong Git. Docker loại `best.pt` khỏi image. Các trọng số `.pt` khác, `.env` và log JSONL được loại khỏi Git qua `.gitignore`.
