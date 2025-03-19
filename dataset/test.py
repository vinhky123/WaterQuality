import pandas as pd

# Đọc file CSV thực tế
df = pd.read_csv("CLN_SG.csv")


# Hàm chuẩn hóa theo tiêu chuẩn WQI giả định (có thể tùy chỉnh theo QCVN)
def normalize_pH(pH):
    if 6.5 <= pH <= 8.5:
        return 100
    elif pH < 6.5:
        return max(0, 100 - (6.5 - pH) * 25)
    else:
        return max(0, 100 - (pH - 8.5) * 25)


def normalize_DO(do):
    if do >= 6:
        return 100
    elif do >= 4:
        return 70
    elif do >= 2:
        return 40
    else:
        return 10


def normalize_NH3(nh3):
    if nh3 <= 0.1:
        return 100
    elif nh3 <= 0.3:
        return 80
    elif nh3 <= 0.5:
        return 60
    elif nh3 <= 1:
        return 40
    else:
        return 20


def normalize_Doduc(turb):
    if turb <= 5:
        return 100
    elif turb <= 10:
        return 80
    elif turb <= 20:
        return 60
    elif turb <= 50:
        return 40
    else:
        return 20


def normalize_TSS(tss):
    if tss <= 20:
        return 100
    elif tss <= 50:
        return 70
    elif tss <= 100:
        return 50
    else:
        return 30


def normalize_Fe(fe):
    if fe <= 0.3:
        return 100
    elif fe <= 0.5:
        return 80
    elif fe <= 1:
        return 60
    else:
        return 40


def normalize_Mn(mn):
    if mn <= 0.1:
        return 100
    elif mn <= 0.3:
        return 80
    elif mn <= 0.5:
        return 60
    else:
        return 40


# Tính điểm từng chỉ số
df["q_pH"] = df["pH_vao_nha_may"].apply(normalize_pH)
df["q_DO"] = df["DO_vao_nha_may"].apply(normalize_DO)
df["q_NH3"] = df["NH3_vao_nha_may"].apply(normalize_NH3)
df["q_Doduc"] = df["Doduc_vao_nha_may"].apply(normalize_Doduc)
df["q_TSS"] = df["SS_vao_nha_may"].apply(normalize_TSS)
df["q_Fe"] = df["Fe_vao_nha_may"].apply(normalize_Fe)
df["q_Mn"] = df["Mn_vao_nha_may"].apply(normalize_Mn)

# Tính WQI trung bình
df["WQI"] = df[["q_pH", "q_DO", "q_NH3", "q_Doduc", "q_TSS", "q_Fe", "q_Mn"]].mean(
    axis=1
)

# Xuất ra file Excel
df.to_csv("WQI_data.csv", index=False)

print("Tính WQI hoàn tất và đã lưu file 'WQI_data.csv'")
