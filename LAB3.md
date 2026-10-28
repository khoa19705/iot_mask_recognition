# Lab 3: Cloud và đo độ trễ

## Phần đã bổ sung

Webcam trên trình duyệt gửi ảnh JPEG đến Flask bằng `POST /api/detect`.
Máy chủ chạy YOLO trên CPU, trả boxes, nhãn, số lượng và thời gian.
Giao diện vẽ kết quả, hiển thị mẫu hiện tại, trung bình và P95, xuất CSV của phiên đo.
Giữ cùng origin cho giao diện và API nên không cần cấu hình CORS hoặc sửa URL API khi chuyển lên Cloud.

`APP_ENV=local` là mặc định. Chỉ đặt `APP_ENV=cloud` trên máy chủ thực sự đã triển khai.
Nhãn môi trường trên giao diện là cấu hình do máy chủ công bố, không tự chứng minh máy chủ đang ở Cloud.
Không cần thêm ESP32-CAM cho bản dùng webcam này.

## Định nghĩa các số đo

| Trường CSV | Điểm bắt đầu và kết thúc |
| --- | --- |
| `inference_ms` | Máy chủ: ngay trước `model.predict()` đến khi predict hoàn tất. Bao gồm tiền xử lý, suy luận và hậu xử lý của YOLO; mẫu đầu có thể gồm khởi tạo predictor. |
| `response_ms` | Trình duyệt: trước gọi fetch đến khi đọc và phân tích xong JSON. Gồm truyền ảnh, thời gian máy chủ, nhận kết quả. Không phải ping hoặc độ trễ mạng thuần. |
| `end_to_end_ms` | Trình duyệt: trước lấy ảnh từ video đến sau cập nhật kết quả và hai callback requestAnimationFrame. Đây là ước lượng cơ hội vẽ lên màn hình, không đo màn hình vật lý hoặc độ trễ cảm biến webcam. |
| `server_ms` | Trong route Flask: đọc request, giải mã ảnh, chờ khóa, nạp model nếu cần, predict, đóng gói detections và ghi log. Không gồm toàn bộ serialization JSON, proxy hoặc thời gian mạng. |
| `queue_ms` | Chờ khóa để chạy model tuần tự trong một worker. |
| `model_load_ms` | Nạp trọng số nếu chưa có model trong RAM. Nếu model đã được `/api/model` tải trước thì bằng 0. |
| `cold_start` | Lần predict đầu tiên của tiến trình máy chủ. |
| `warmup` | Ba mẫu đầu của phiên đo hoặc mẫu cold start, giữ trong CSV nhưng bỏ khỏi trung bình/P95. |

Thời gian trên máy chủ và trình duyệt đều đo bằng đồng hồ đơn điệu của chính máy đó.
Không trừ timestamp của hai máy; không yêu cầu đồng bộ đồng hồ để tính các khoảng thời gian này.
P95 dùng nearest rank: phần tử thứ `ceil(0.95*N)` trong dãy tăng dần.
Khoảng nghỉ 600 ms giữa các yêu cầu không nằm trong độ trễ một ảnh. Không suy ra FPS video từ `1000 / inference_ms`.
Chuyển tab nền sẽ tạm dừng lấy mẫu; mẫu đang xử lý khi tab ẩn không đưa vào thống kê.
Tối đa 10.000 mẫu gần nhất được giữ trong RAM của tab. CSV không tự lưu lên Cloud và sẽ mất nếu tải lại trang.

## Chạy local

Mở terminal tại thư mục `mask_detect`:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Mở http://localhost:5000, cấp quyền camera, bật webcam. Server mới cần được khởi động lại sau sửa Python.
Tắt webcam rồi bấm **Phiên đo mới** để reset dữ liệu. Tải CSV trước khi đóng trang.
Không đưa `.env` lên Git. `.env.example` chứa cấu hình mẫu.

## Triển khai Docker

Docker chạy Linux với Gunicorn, một worker và bốn thread; khóa riêng tuần tự hóa YOLO.
Một worker giúp không nhân bản bộ nhớ model và giữ log trong cùng tiến trình.
File Docker cài PyTorch CPU từ nguồn chính thức. Các dependency vẫn dùng khoảng phiên bản trong requirements; lần build sau có thể lấy bản mới.

Nếu máy có Docker, từ thư mục `mask_detect`:

```powershell
docker build -t masklens-lab3 .
docker run --rm -p 10000:10000 -e PORT=10000 -e APP_ENV=local masklens-lab3
```

Mở http://localhost:10000. Sau đó có thể đưa image lên dịch vụ chạy Docker với HTTPS và đặt `APP_ENV=cloud`.
Docker build tại máy này chưa được xác minh nếu chưa cài Docker. Chạy container local vẫn chưa phải Cloud.

## Cấu hình Render hoặc dịch vụ Docker tương đương

Tài liệu chính thức:
- https://render.com/docs/docker
- https://render.com/docs/health-checks
- https://flask.palletsprojects.com/en/stable/deploying/gunicorn/

1. Bạn đăng nhập dịch vụ Cloud và kết nối repository của ứng dụng. Không có thao tác tạo tài khoản hoặc mua tài nguyên nào được tự thực hiện.
2. Nếu repo là `mask_detect`, dùng Dockerfile và `render.yaml` ngay gốc repo. Nếu repo là `Iot`, đặt Root Directory là `mask_detect`; Blueprint cũng cần trỏ đúng file trong thư mục đó.
3. Đặt `APP_ENV=cloud`, `FLASK_DEBUG=0`, `MAX_UPLOAD_MB=8`; dịch vụ cung cấp `PORT`. Health check: `/api/health`.
4. Cung cấp model bằng một trong hai cách:
   - Build image từ bộ ZIP có `best.pt` rồi đưa image lên registry/dịch vụ hỗ trợ image. ZIP có sẵn model.
   - Repo hiện tại đã chứa `best.pt` nên build từ Git dùng trực tiếp trọng số, không cần `MODEL_URL`. Nếu sau này tách trọng số khỏi Git, có thể đặt `MODEL_URL` HTTPS tải trực tiếp và `MODEL_SHA256`; script sẽ kiểm tra hash trước khi dùng.
5. Lấy SHA256 của trọng số:

```powershell
(Get-FileHash -LiteralPath .\best.pt -Algorithm SHA256).Hash.ToLower()
```

6. Sau deploy, mở URL HTTPS của Cloud. Trình duyệt vẫn dùng webcam laptop, ảnh được gửi tới máy chủ Cloud cùng origin. Đợi thông tin đủ 3 nhóm hiện lên rồi chạy thử.
7. Kiểm tra `GET /api/model`: ba nhãn phải là `mask_weared_incorrect`, `with_mask`, `without_mask`; `three_classes_ready` phải true.

Model hiện tại khoảng 52 MB và 25,9 triệu tham số. Chưa đủ bằng chứng để gọi đây là model nhẹ theo tiêu chí của giảng viên.
Ứng dụng còn cần RAM cho PyTorch, mô hình và ảnh; cần chọn cấu hình theo mức dùng thực tế, không mặc định gói nhỏ nhất sẽ chạy được.
Nếu cần model nhỏ hơn, có thể huấn luyện lại bằng `train.py --model yolo11n.pt --data <dataset.yaml>`; hiện project chưa chứa dataset để tự huấn luyện hoặc đánh giá accuracy.

Model chỉ được kiểm tra cấu trúc, tải và suy luận; cần thử ảnh thực tế để đánh giá độ chính xác.
Log phát hiện trên filesystem container có thể mất khi restart/redeploy. Với demo hãy tải log/CSV về, hoặc gắn persistent storage tại `/app/logs` nếu muốn giữ lịch sử.
Ứng dụng hiện là demo không có đăng nhập; ai truy cập URL cũng có thể gọi API và đọc log phát hiện chung. Chỉ sử dụng với dữ liệu demo và thu hồi dịch vụ sau buổi lab hoặc thêm kiểm soát truy cập trước khi dùng rộng rãi.

## Kịch bản kiểm thử để báo cáo

1. Chạy local, ghi phiên bản model, cấu hình máy và độ phân giải ảnh gửi.
2. Thử đeo đúng, không đeo, đeo chưa đúng; sau đó có nhiều người thuộc các nhóm khác nhau trong cùng khung hình. Ghi lại trường hợp sai, không chỉ trường hợp đúng.
3. Giữ tab hiển thị và thu ít nhất 100 mẫu thống kê sau warmup; tải CSV local.
4. Lặp lại trên URL Cloud với cùng model, ngưỡng tin cậy và điều kiện camera; ghi cấu hình Cloud, kết nối mạng và vị trí máy chủ nếu biết. Tải CSV Cloud.
5. Dùng trung bình/P95 trên giao diện hoặc CSV để so sánh ba loại thời gian. Thời gian khởi động có sẵn trong CSV để bàn riêng, không gọi đó là tốc độ ổn định.
6. Chụp màn hình nhận diện nhiều nhãn và bảng độ trễ trên URL Cloud. Bảng thống kê ít mẫu chỉ là minh họa, không chứng minh hiệu năng chung.

Đề xuất 100 mẫu và kịch bản trên là hướng dẫn kiểm thử, không phải số lượng bắt buộc ghi trong slide.

## Kiểm tra tự động và đóng gói

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/metrics.test.js
node --test tests/frontend.test.js
node --check static/js/app.js
python verify_model.py
python package_cloud.py
```

`verify_model.py` chạy model thật trên 5 ảnh trống 640×480, tạo `outputs/model-verification.json`.
Đây là smoke test CPU, không phải kiểm thử accuracy, webcam hay độ trễ Cloud.
`package_cloud.py` tạo `outputs/masklens-lab3-cloud.zip` theo danh sách file cho phép, có model, không có `.env` và log cá nhân.

## Trạng thái kiểm tra ngày 28/10/2026

- 8 kiểm thử Python API/tải model và 5 kiểm thử JavaScript thống kê/luồng giao diện mô phỏng: đạt.
- Model thật tải thành công, đủ 3 nhãn; suy luận trên 5 ảnh trống: đạt. Các mẫu này không dùng để đánh giá độ chính xác.
- Giao diện được mở và kiểm tra bố cục trên trình duyệt; chưa hoàn tất thử webcam với người thật.
- Docker chưa được build vì môi trường hiện tại chưa có Docker. Chưa triển khai Cloud, chưa có số liệu độ trễ Cloud.
- Bộ ZIP được đóng gói từ danh sách file cho phép, chứa trọng số, không chứa `.env` hoặc log phát hiện.
