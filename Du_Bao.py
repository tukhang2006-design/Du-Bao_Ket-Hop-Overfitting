import csv
import math
import random


# 1. ĐỌC FILE CSV VÀ CHUẨN BỊ DỮ LIỆU

def load_csv(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            if row:
                data.append([float(val) for val in row])
    return headers, data

csv_filename = "Du_Bao_Gia_Nha.csv"
print(f"[*] Đang đọc file '{csv_filename}'...")
headers, data = load_csv(csv_filename)
print(f"[+] Đọc thành công! Tổng cộng {len(data)} dòng dữ liệu.")

# Xáo trộn và chia dữ liệu: 75% Train, 25% Test
random.seed(42)
shuffled_data = list(data)
random.shuffle(shuffled_data)

split_idx = int(len(shuffled_data) * 0.75)
train_data = shuffled_data[:split_idx]
test_data = shuffled_data[split_idx:]


# 2. XÂY DỰNG MÔ HÌNH CÂY QUYẾT ĐỊNH

class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def calc_variance(y):
    if len(y) <= 1:
        return 0.0
    mean = sum(y) / len(y)
    return sum((val - mean) ** 2 for val in y)

def find_best_split(dataset, min_samples_leaf):
    best_cost = float('inf')
    best_feat, best_thresh, best_left, best_right = None, None, None, None
    n_features = len(dataset[0]) - 1  # Cột cuối là nhãn giá nhà

    for feat_idx in range(n_features):
        vals = sorted(list(set(row[feat_idx] for row in dataset)))
        # Lấy các mốc chia mẫu đại diện để tính toán nhanh
        step = max(1, len(vals) // 20)
        candidates = vals[::step]

        for thresh in candidates:
            left = [row for row in dataset if row[feat_idx] <= thresh]
            right = [row for row in dataset if row[feat_idx] > thresh]

            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue

            cost = calc_variance([r[-1] for r in left]) + calc_variance([r[-1] for r in right])
            if cost < best_cost:
                best_cost = cost
                best_feat = feat_idx
                best_thresh = thresh
                best_left = left
                best_right = right

    return best_feat, best_thresh, best_left, best_right

def build_tree(dataset, depth=0, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    y = [row[-1] for row in dataset]
    mean_val = sum(y) / len(y)

    # Điều kiện dừng
    if len(dataset) < min_samples_split:
        return TreeNode(value=mean_val)
    if max_depth is not None and depth >= max_depth:
        return TreeNode(value=mean_val)
    if len(set(y)) == 1:
        return TreeNode(value=mean_val)

    feat, thresh, left, right = find_best_split(dataset, min_samples_leaf)
    if feat is None:
        return TreeNode(value=mean_val)

    left_node = build_tree(left, depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right_node = build_tree(right, depth + 1, max_depth, min_samples_split, min_samples_leaf)

    return TreeNode(feature_idx=feat, threshold=thresh, left=left_node, right=right_node, value=mean_val)

def predict_one(node, row):
    if node.feature_idx is None:
        return node.value
    if row[node.feature_idx] <= node.threshold:
        return predict_one(node.left, row)
    return predict_one(node.right, row)

def predict(tree, dataset):
    return [predict_one(tree, row) for row in dataset]

# Các hàm đo lường hiệu suất
def calc_r2(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    return 1 - (ss_res / ss_tot)

def calc_rmse(y_true, y_pred):
    mse = sum((y - p) ** 2 for y, p in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


# 3. HUẤN LUYỆN: MÔ HÌNH OVERFITTING vs MÔ HÌNH REGULARIZATION (TỈA CÂY)

print("[*] Đang huấn luyện mô hình 1...")
# Mô hình 1: Cho cây mọc tự do tới từng mẫu dữ liệu nhỏ nhất
overfit_tree = build_tree(train_data, max_depth=None, min_samples_split=2, min_samples_leaf=1)

print("[*] Đang huấn luyện mô hình 2...")
# Mô hình 2: Giới hạn độ sâu 4 tầng, mỗi lá phải có ít nhất 8 căn nhà
regularized_tree = build_tree(train_data, max_depth=4, min_samples_split=15, min_samples_leaf=8)

# Lấy nhãn thực tế
y_train = [row[-1] for row in train_data]
y_test = [row[-1] for row in test_data]

# Dự báo mô hình Overfit
pred_train_of = predict(overfit_tree, train_data)
pred_test_of = predict(overfit_tree, test_data)
r2_train_of = calc_r2(y_train, pred_train_of)
r2_test_of = calc_r2(y_test, pred_test_of)
rmse_test_of = calc_rmse(y_test, pred_test_of)

# Dự báo mô hình Regularized
pred_train_reg = predict(regularized_tree, train_data)
pred_test_reg = predict(regularized_tree, test_data)
r2_train_reg = calc_r2(y_train, pred_train_reg)
r2_test_reg = calc_r2(y_test, pred_test_reg)
rmse_test_reg = calc_rmse(y_test, pred_test_reg)


# 4. KẾT QUẢ SO SÁNH

print("\n" + "=" * 68)
print(f"{'MÔ HÌNH':<28} | {'TRAIN R²':<10} | {'TEST R²':<10} | {'TEST RMSE (Triệu)':<15}")
print("=" * 68)
print(f"{'1. Overfit (Không tỉa)':<28} | {r2_train_of:<10.4f} | {r2_test_of:<10.4f} | {rmse_test_of:<15.2f}")
print(f"{'2. Regularized (Tỉa cây)':<28} | {r2_train_reg:<10.4f} | {r2_test_reg:<10.4f} | {rmse_test_reg:<15.2f}")
print("=" * 68)

print("\n[*] PHÂN TÍCH HIỆN TƯỢNG OVERFITTING & KHẮC PHỤC:")
print(f"- Mô hình Overfit: Điểm Train R² đạt tuyệt đối {r2_train_of*100:.1f}%, nhưng Test R² sụt xuống {r2_test_of*100:.2f}%.")
print(f"- Mô hình Regularized: Sau khi tỉa cành, Test R² tăng lên {r2_test_reg*100:.2f}%, Test RMSE giảm từ {rmse_test_of:.2f} xuống {rmse_test_reg:.2f} triệu VNĐ.")
print(f"=> Khoảng cách Train - Test thu hẹp từ {(r2_train_of - r2_test_of)*100:.2f}% xuống còn {(r2_train_reg - r2_test_reg)*100:.2f}%.")