import os
from glob import glob
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import seaborn as sns
from math import pi
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(layout="wide") # Mở rộng giao diện

@st.cache_data
def load_transaction_info(file_path):
    """Hàm đọc file transaction_info có sử dụng cache để tránh giật lag"""
    if not os.path.exists(file_path):
        return [], False, None
    try:
        df_transaction = pl.read_excel(file_path)
        if "campaignID" in df_transaction.columns:
            df_transaction = df_transaction.with_columns(pl.col("campaignID").cast(pl.String))
            list_ids = df_transaction.drop_nulls("campaignID")["campaignID"].unique().sort().to_list()
            return list_ids, True, df_transaction
        return [], False, df_transaction
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return [], False, None
def calculate_metrics(df: pl.DataFrame):
    """Tính toán KPIs, có thêm logic bắt lỗi phòng hờ dữ liệu trống"""
    if df is None or df.is_empty() or 'total_npu' not in df.columns:
        return 0, 0, 0, 0, 0

    # Dùng Polars chuẩn để lấy giá trị scalar
    total_npu = df.select(pl.sum("total_npu")).item() or 0
    total_revenue = df.select(pl.sum("total_revenue")).item() or 0
    total_cost = df.select(pl.sum("total_cost")).item() or 0
    
    cac = (total_cost / total_npu) if total_npu > 0 else 0
    roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    return total_npu, total_revenue, total_cost, cac, roi

import polars as pl
import pandas as pd

def calculate_advanced_metrics(df_trans: pl.DataFrame, target_col: str = "platform") -> pd.DataFrame:
    """
    Hàm tính toán các chỉ số gom nhóm theo Tháng và Đối tượng (Platform hoặc sof) để vẽ biểu đồ.
    
    Args:
        df_trans (pl.DataFrame): Dataframe gốc chứa dữ liệu giao dịch.
        target_col (str): Tên cột muốn dùng làm 'Đối tượng' ('platform' hoặc 'sof'). Mặc định là 'platform'.
    """
    if df_trans.is_empty():
        return pd.DataFrame() # Trả về DF rỗng nếu không có dữ liệu
        
    # Chuyển đổi chuỗi ngày tháng sang datetime nếu cần
    if df_trans.schema["reqDate"] == pl.String:
        df_trans = df_trans.with_columns(
            pl.col("reqDate").str.to_datetime("%Y-%m-%d %H:%M:%S.%f", strict=False)
        )
    
    # Lọc các giao dịch thành công
    df_success = df_trans.filter(pl.col("transStatus") == 1)
    
    if df_success.is_empty():
        return pd.DataFrame()

    # Tạo cột Thời gian và Đối tượng dựa trên tham số target_col
    df_prepared = df_success.with_columns([
        pl.col("reqDate").dt.strftime("%Y-%m").alias("Thời gian"),
        pl.col(target_col).fill_null("Unknown").alias("Đối tượng")
    ])
    
    # Tính toán metrics
    df_aggregated = df_prepared.group_by(["Thời gian", "Đối tượng"]).agg([
        pl.col("amount").sum().alias("Doanh thu"),
        pl.col("discountAmount").sum().alias("Tổng Khuyến mãi"),
        pl.col("transID").count().alias("Số giao dịch"),
        pl.col("userID").n_unique().alias("Số User unique")
    ]).sort("Thời gian")
    
    return df_aggregated.to_pandas() # Convert sang Pandas cho Plotly dễ vẽ
def create_summary_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Tạo bảng thống kê tổng hợp từ bảng giao dịch"""
    if df is None or df.is_empty():
        return pl.DataFrame()

    df_summary = (
        df.filter(pl.col("transStatus") == 1)
        .group_by("campaignID")
        .agg([
            pl.col("userID").n_unique().alias("total_npu"),
            pl.col("userChargeAmount").sum().fill_null(0).alias("total_revenue"),
            pl.col("discountAmount").sum().fill_null(0).alias("total_cost"),
        ])
        .with_columns([
            # Tính CAC (Cost Per Acquisition) - Xử lý lỗi chia cho 0
            pl.when(pl.col("total_npu") > 0)
            .then(pl.col("total_cost") / pl.col("total_npu"))
            .otherwise(0)
            .alias("cac"),
            
            # Tính ROI (Return on Investment) - Xử lý lỗi chia cho 0
            pl.when(pl.col("total_cost") > 0)
            .then(((pl.col("total_revenue") - pl.col("total_cost")) / pl.col("total_cost")) * 100)
            .otherwise(0)
            .alias("roi")
        ])
        # Sắp xếp theo doanh thu giảm dần
        .sort("total_revenue", descending=True)
    )
    return df_summary
# =================================================================
# --- ĐỌC DỮ LIỆU & SIDEBAR ---
# =================================================================
st.sidebar.header("🔍 Bộ lọc hệ thống")

# Đọc file transaction
trans_path = "data/transaction_info.xlsx"
list_ids, is_success, trans_df = load_transaction_info(trans_path)
campaign_summary_df = create_summary_dataframe(trans_df) if is_success else pl.DataFrame()
if not is_success:
    st.sidebar.warning("⚠️ Không tìm thấy file 'transaction_info.xlsx' hoặc bị lỗi cấu trúc.")

# Quản lý State của multiselect
if 'selected_campaign_ids' not in st.session_state:
    st.session_state['selected_campaign_ids'] = []  

def select_all():
    st.session_state['selected_campaign_ids'] = list_ids

def clear_all():
    st.session_state['selected_campaign_ids'] = []
def add_random_10_campaign_id():
    if list_ids:
        current_selection = set(st.session_state['selected_campaign_ids'])
        remaining_ids = list(set(list_ids) - current_selection)
        if remaining_ids:
            random_selection = np.random.choice(remaining_ids, size=min(10, len(remaining_ids)), replace=False)
            st.session_state['selected_campaign_ids'] = list(current_selection.union(random_selection))

# Giao diện bộ lọc Sidebar
if list_ids:
    st.sidebar.button("✅ Chọn tất cả", on_click=select_all, use_container_width=True)
    st.sidebar.button("🗑️ Xóa hết", on_click=clear_all, use_container_width=True)
    st.sidebar.button("🎲 Thêm 10 campaign ngẫu nhiên", on_click=add_random_10_campaign_id, use_container_width=True)

    Selected_merchants = st.sidebar.multiselect(    
        "Tùy chỉnh campaignID:", 
        options=list_ids,
        key='selected_campaign_ids' 
    )
    
    # --- LOGIC LỌC DỮ LIỆU CHÍNH ---
    if Selected_merchants: 
        # Bước 1: Lọc data thô theo CampaignID
        df_filtered_raw = trans_df.filter(pl.col("campaignID").cast(pl.String).is_in(Selected_merchants))
        
        # Bước 2: Đưa data đã lọc đi tính toán để vẽ biểu đồ
        chart_df = calculate_advanced_metrics(df_filtered_raw,target_col="platform")
        chart1_df = calculate_advanced_metrics(df_filtered_raw,target_col="sof")
        status_thanh_cong = not chart_df.empty
    else:
        st.warning("👈 Vui lòng chọn hoặc thêm ít nhất 1 campaignID ở thanh bên trái để xem dữ liệu.")
        df_filtered_raw = pl.DataFrame()
        chart_df = pd.DataFrame()
        status_thanh_cong = False
else:
    Selected_merchants = []
    df_filtered_raw = pl.DataFrame()
    chart_df = pd.DataFrame()
    status_thanh_cong = False
    df = calculate_advanced_metrics(df_filtered_raw)
# =================================================================
# --- GIAO DIỆN HIỂN THỊ CHÍNH ---
# =================================================================


# 2. CHỌN TIÊU CHÍ VẼ BIỂU ĐỒ
st.markdown("### 📊 TỔNG QUAN THỊ TRƯỜNG:")
tieu_chi_duoc_chon = st.selectbox(
    "Chọn tiêu chí muốn xem trên biểu đồ:",
    options=['Doanh thu', 'Tổng Khuyến mãi', 'Số giao dịch', 'Số User unique']
)




# 1. Trích xuất danh sách các campaignID duy nhất đang có mặt trong df_filtered_raw
valid_campaigns = df_filtered_raw.select(pl.col("campaignID").cast(pl.String)).unique().to_series().to_list()

# 2. Lọc campaign_summary_df
filtered_campaign_summary = campaign_summary_df.filter(pl.col("campaignID").cast(pl.String).is_in(valid_campaigns))
with st.container():
    total_npu, total_revenue, total_cost, cac, roi = calculate_metrics(filtered_campaign_summary)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="👥 Tổng số giao dịch", value=f"{total_npu:,}")
    with col2:
        st.metric(label="💰 Tổng Doanh thu", value=f"{total_revenue:,}")
    with col3:
        st.metric(label="💸 Tổng Chi phí", value=f"{total_cost:,}")
    with col4:
        st.metric(label="📊 CAC", value=f"{cac:,.2f}")
    with col5:
        st.metric(label="📈 ROI", value=f"{roi:,.2f}%")

st.write("") 
# 3. BIỂU ĐỒ
with st.container():
    left_col, right_col = st.columns([2, 1])
    
    # --- CỘT TRÁI (Biểu đồ chính) ---
    with left_col:
        danh_sach_thang = chart_df['Thời gian'].unique().tolist() if status_thanh_cong else []
        st.subheader(f"{tieu_chi_duoc_chon.upper()} THEO NỀN TẢNG (THÁNG **{int(danh_sach_thang[0].split('-')[1])}** - THÁNG **{int(danh_sach_thang[-1].split('-')[1])}**)")
        
        if not status_thanh_cong or chart_df.empty:
            st.info("Chưa có dữ liệu biểu đồ. Hãy chọn CampaignID có giao dịch thành công.")
        else:
            
            
            # Slider chọn tháng
            start_thang, end_thang = st.select_slider(
                "Chọn khoảng thời gian hiển thị (Cột):",
                options=danh_sach_thang,
                value=(danh_sach_thang[0], danh_sach_thang[-1])
            )

            start_idx = danh_sach_thang.index(start_thang)
            end_idx = danh_sach_thang.index(end_thang)
            thang_duoc_chon = danh_sach_thang[start_idx : end_idx + 1]

            chart_df_filtered = chart_df[chart_df['Thời gian'].isin(thang_duoc_chon)]

            fig_bar = px.bar(
                chart_df_filtered, 
                x='Thời gian', 
                y=tieu_chi_duoc_chon, 
                color='Đối tượng', 
                barmode='group', 
                color_discrete_sequence=px.colors.qualitative.Set2 # Dùng màu động
            )
            
            fig_bar.update_layout(
                xaxis_title="",
                yaxis_title="Giá trị",
                legend_title="Đối tượng",
                hovermode="x unified",
                margin=dict(t=20, b=20, l=0, r=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
    # --- CỘT PHẢI (Biểu đồ cơ cấu phụ) ---
    with right_col:
        st.subheader(f"CƠ CẤU **{tieu_chi_duoc_chon.upper()}** THEO NGUỒN TIỀN (SOF)")
        
        if status_thanh_cong and not chart1_df.empty:
            thoi_gian_list = chart1_df['Thời gian'].unique()
            thoi_gian_chon = st.select_slider(
                "Lựa chọn Tháng phân tích (Tròn):",
                options=thoi_gian_list
            )
            st.markdown("<br>", unsafe_allow_html=True) # Tạo khoảng trống cho cân đối với cột trái
            
            chart_df_pie_filtered = chart1_df[chart1_df['Thời gian'] == thoi_gian_chon]

            fig_pie = px.pie(
                chart_df_pie_filtered, 
                values=tieu_chi_duoc_chon, 
                names='Đối tượng',
                hole=0.5, 
                color='Đối tượng',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig_pie.update_traces(
                textinfo='percent+value', 
                textfont_size=14,
                hoverinfo='label+percent+value'
            )
            
            fig_pie.update_layout(
                annotations=[dict(text=tieu_chi_duoc_chon.split()[0], x=0.5, y=0.5, font_size=16, showarrow=False)],
                legend_title="Đối tượng",
                margin=dict(t=20, b=20, l=0, r=0)
            )

            st.plotly_chart(fig_pie, use_container_width=True)


st.markdown("---")














































def create_summary_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    df_summary = (
        df.filter(pl.col("transStatus") == 1)
        .group_by("campaignID")
        .agg([
            pl.col("userID").n_unique().alias("total_npu"),
            pl.len().alias("total_transactions"),
            pl.col("userChargeAmount").sum().fill_null(0).alias("total_revenue"),
            pl.col("discountAmount").sum().fill_null(0).alias("total_cost"),
        ])
        .with_columns([
            pl.when(pl.col("total_npu") > 0)
            .then(pl.col("total_cost") / pl.col("total_npu"))
            .otherwise(0)
            .alias("cac"),
            
            pl.when(pl.col("total_cost") > 0)
            .then(((pl.col("total_revenue") - pl.col("total_cost")) / pl.col("total_cost")) * 100)
            .otherwise(0)
            .alias("roi")
        ])
        .sort("total_revenue", descending=True)
    )
    return df_summary

def calculate_advanced_metrics(df_trans: pl.DataFrame) -> pl.DataFrame:
    if df_trans.schema["reqDate"] == pl.String:
        df_trans = df_trans.with_columns(
            pl.col("reqDate").str.to_datetime("%Y-%m-%d %H:%M:%S.%f")
        )
        
    df_fraud_devices = (
        df_trans.group_by("deviceID")
        .agg(pl.col("userID").n_unique().alias("users_per_device"))
        .filter(pl.col("users_per_device") >= 3)
    )
    
    df_trans = df_trans.join(df_fraud_devices, on="deviceID", how="left").with_columns(
        pl.col("users_per_device").is_not_null().alias("is_fraud_user")
    )

    df_success = df_trans.filter(pl.col("transStatus") == 1).sort(["userID", "reqDate"])
    df_success = df_success.with_columns(
        pl.col("reqDate").first().over(["campaignID", "userID"]).alias("first_reqDate")
    )
    df_success = df_success.with_columns(
        (pl.col("reqDate") - pl.col("first_reqDate")).dt.total_days().alias("days_since_first")
    )

    df_success = df_success.with_columns([
        pl.when((pl.col("days_since_first") > 0) & (pl.col("days_since_first") <= 30))
        .then(1).otherwise(0).alias("is_M1"),
        
        pl.when((pl.col("days_since_first") >= 61) & (pl.col("days_since_first") <= 90))
        .then(1).otherwise(0).alias("is_M3"),
        
        pl.when(pl.col("days_since_first") > 0).then(1).otherwise(0).alias("is_subsequent_trans"),
        
        pl.when((pl.col("days_since_first") > 0) & 
                ((pl.col("discountAmount") > 0) | pl.col("promotion_type").is_not_null()))
        .then(1).otherwise(0).alias("is_subsequent_promo"),
    ])

    df_summary_success = (
        df_success.group_by("campaignID").agg([
            pl.col("userID").n_unique().alias("total_npu"),
            pl.col("userID").filter(pl.col("is_M1") == 1).n_unique().alias("users_M1"),
            pl.col("userID").filter(pl.col("is_M3") == 1).n_unique().alias("users_M3"),
            pl.col("is_subsequent_trans").sum().alias("total_subsequent_trans"),
            pl.col("is_subsequent_promo").sum().alias("total_subsequent_promo"),
        ])
    )

    df_summary_fraud = (
        df_trans.group_by("campaignID").agg([
            pl.col("userID").n_unique().alias("total_users_all"),
            pl.col("userID").filter(pl.col("is_fraud_user")).n_unique().alias("fraud_users"),
        ])
    )

    df_final = (
        df_summary_success.join(df_summary_fraud, on="campaignID", how="outer")
        .with_columns([
            ((pl.col("users_M1") / pl.col("total_npu")) * 100).fill_null(0).round(2).alias("R1M"),
            ((pl.col("users_M3") / pl.col("total_npu")) * 100).fill_null(0).round(2).alias("R3M"),
            pl.when(pl.col("total_subsequent_trans") > 0)
            .then((pl.col("total_subsequent_promo") / pl.col("total_subsequent_trans")) * 100)
            .otherwise(0).round(2).alias("Promo_Dependency_PCT"),
            ((pl.col("fraud_users") / pl.col("total_users_all")) * 100).fill_null(0).round(2).alias("Fraud_Rate_PCT")
        ])
        .select(["campaignID", "R1M", "R3M", "Promo_Dependency_PCT", "fraud_users", "Fraud_Rate_PCT"])
    )
    return df_final

def create_base_dataframe(campaign_summary_df, metric_summary_df):
    merged_df = pd.merge(
        campaign_summary_df.to_pandas(), 
        metric_summary_df.to_pandas(), 
        on='campaignID', 
        how='inner'
    )
    
    merged_df['Avg_Freq'] = np.where(
        merged_df['total_npu'] > 0, 
        merged_df['total_transactions'] / merged_df['total_npu'], 
        0
    )
    
    rename_mapping = {
        'total_npu': 'NPU',
        'total_revenue': 'Doanh thu',
        'total_cost': 'Khuyến mãi',
        'cac': 'CAC',
        'roi': 'ROI_Pct'
    }
    
    df_renamed = merged_df.rename(columns=rename_mapping)
    target_columns = [
        'campaignID', 'NPU', 'Doanh thu', 'Khuyến mãi', 
        'CAC', 'ROI_Pct', 'R1M', 'Avg_Freq', 'Fraud_Rate_PCT'
    ]
    
    df_base = df_renamed[target_columns].copy()
    df_base['Avg_Freq'] = df_base['Avg_Freq'].round(2)
    return df_base

def calculate_campaign_scores(df_base: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    df = df_base.copy()
    positive_metrics = ['NPU', 'Doanh thu', 'ROI_Pct', 'R1M', 'Avg_Freq']
    negative_metrics = ['Khuyến mãi', 'CAC', 'Fraud_Rate_PCT']
    
    if weights is None:
        weights = {
            'NPU_Score': 0.10,
            'Doanh thu_Score': 0.10,
            'Khuyến mãi_Score': 0.10,
            'CAC_Score': 0.10,
            'ROI_Pct_Score': 0.10,
            'R1M_Score': 0.30,
            'Avg_Freq_Score': 0.10,
            'Fraud_Rate_PCT_Score': 0.10
        }

    def min_max_scale(series, is_positive=True):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(100.0, index=series.index)
        if is_positive:
            return ((series - min_val) / (max_val - min_val)) * 100
        else:
            return ((max_val - series) / (max_val - min_val)) * 100

    for col in positive_metrics:
        if col in df.columns: df[f'{col}_Score'] = min_max_scale(df[col], is_positive=True)
    for col in negative_metrics:
        if col in df.columns: df[f'{col}_Score'] = min_max_scale(df[col], is_positive=False)

    df['Total_Score'] = 0.0
    for score_col, weight in weights.items():
        if score_col in df.columns:
            df['Total_Score'] += df[score_col] * weight

    score_columns = [col for col in df.columns if '_Score' in col]
    df[score_columns] = df[score_columns].round(2)
    df = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
    return df


# ==========================================
# MAIN DASHBOARD 
# ==========================================
st.title("📊 Dashboard Đánh Giá Chất Lượng Campaign (Top 10)")
st.markdown("Ứng dụng sử dụng framework chấm điểm (Scoring Framework) dựa trên phương pháp chuẩn hóa **Min-Max** để đánh giá toàn diện hiệu quả của các chiến dịch Marketing.")

# Bước 1: Load Dữ Liệu
trans_path = "data/transaction_info.xlsx"
list_ids, is_success, trans_df = load_transaction_info(trans_path)

# Kiểm tra dữ liệu đầu vào. Nếu thất bại, báo lỗi và dừng toàn bộ app.
if not is_success or trans_df is None:
    st.error(f"❌ Không tìm thấy file excel tại đường dẫn: `{trans_path}` hoặc dữ liệu không hợp lệ. Vui lòng kiểm tra lại!")
    st.stop() # Dừng chạy các mã bên dưới

st.success("✅ Đã load dữ liệu thành công từ file thực tế.")

# ---------------------------------------------------------
# XỬ LÝ DỮ LIỆU & LỌC TOP 10 THEO TOTAL SCORE
# ---------------------------------------------------------
campaign_summary_df = create_summary_dataframe(df_filtered_raw)
metric_summary_df = calculate_advanced_metrics(df_filtered_raw)
df_base = create_base_dataframe(campaign_summary_df, metric_summary_df)

# Tính điểm toàn bộ, sau đó LẤY NGAY TOP 10 CAMPAIGN XUẤT SẮC NHẤT
df_all_scores = calculate_campaign_scores(df_base)
df_final = df_all_scores.head(10)

# Lấy danh sách ID của Top 10 để đồng bộ cho biểu đồ Heatmap & Dữ liệu giao dịch
top_10_ids = df_final['campaignID'].tolist()

# Lọc dữ liệu Transaction gốc chỉ giữ lại thông tin của Top 10 Campaign này
df_transactions = df_filtered_raw.filter(pl.col("campaignID").is_in(top_10_ids))

# Cấu hình danh sách các cột điểm để vẽ Radar Chart
score_cols = [col for col in df_final.columns if '_Score' in col and col != 'Total_Score']

st.subheader("🗂️ Bảng xếp hạng và Dữ liệu chi tiết (Top 10)")
st.dataframe(df_final.style.format(precision=2).background_gradient(subset=['Total_Score'], cmap='Greens'), use_container_width=True)

st.divider()

# Bước 2: Vẽ Biểu Đồ Radar & Tổng điểm
st.subheader("📈 Trực quan hóa dữ liệu")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("**1. Xếp hạng Tổng điểm (Total Score)**")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(x='campaignID', y='Total_Score', data=df_final, palette='viridis', ax=ax_bar)
    ax_bar.set_ylabel('Điểm số (Thang 100)')
    ax_bar.set_xlabel('Campaign')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(df_final['Total_Score']):
        ax_bar.text(i, v + 1, str(round(v, 1)), ha='center', fontweight='bold', fontsize=9)
    st.pyplot(fig_bar)

with col2:
    st.markdown("**2. So sánh điểm thành phần (Radar Chart)**")
    top_3_campaigns = df_final['campaignID'].head(3).tolist()
    selected_campaigns = st.multiselect(
        "Chọn các Campaign để so sánh:",
        options=df_final['campaignID'].tolist(),
        default=top_3_campaigns
    )
    
    def create_radar_chart(df, campaigns_to_plot, score_columns):
        if not campaigns_to_plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.text(0.5, 0.5, 'Vui lòng chọn ít nhất 1 Campaign', ha='center', va='center')
            ax.axis('off')
            return fig

        N = len(score_columns)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        labels = [col.replace('_Score', '') for col in score_columns]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        
        ax.set_rlabel_position(0)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=7)
        ax.set_ylim(0, 100)
        
        for idx, row in df[df['campaignID'].isin(campaigns_to_plot)].iterrows():
            values = row[score_columns].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['campaignID'])
            ax.fill(angles, values, alpha=0.1)
            
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        return fig

    fig_radar = create_radar_chart(df_final, selected_campaigns, score_cols)
    st.pyplot(fig_radar)

st.divider()

# ==========================================
# BƯỚC 3: PHÂN TÍCH CHUYÊN SÂU TỪNG CHỈ SỐ
# ==========================================
st.write("### 🔍 Phân tích chi tiết và So sánh các chỉ số của Top 10 Campaign")

# 1. Tuỳ chọn các chỉ số để ném vào bảng DataFrame
available_metrics = ['NPU', 'Doanh thu', 'Khuyến mãi', 'CAC', 'ROI_Pct', 'R1M', 'Avg_Freq', 'Fraud_Rate_PCT']

selected_metrics = st.multiselect(
    "Tùy chọn các chỉ số muốn hiển thị và so sánh trong bảng tổng hợp Top 10:",
    options=available_metrics,
    default=['Doanh thu', 'NPU', 'ROI_Pct', 'Khuyến mãi']
)

if selected_metrics:
    # Cắt lọc các cột từ df_final theo lựa chọn của người dùng
    display_cols = ['campaignID'] + selected_metrics
    df_top_10_summary = df_final[display_cols]

    st.write("#### Bảng so sánh chỉ số tùy chỉnh:")
    st.dataframe(df_top_10_summary.style.format(precision=2), use_container_width=True)

    # 2. Biểu đồ Bar Chart động dựa trên lựa chọn
    st.write("#### Biểu đồ trực quan hóa chỉ số")
    selected_plot_metric = st.selectbox(
        "Chọn 1 chỉ số từ danh sách bạn vừa tạo để vẽ biểu đồ so sánh:", 
        options=selected_metrics
    )
    
    # ÉP KIỂU campaignID SANG CHUỖI (STR) ĐỂ PLOTLY HIỂU LÀ CATEGORY
    df_plot = df_top_10_summary.copy()
    df_plot['campaignID'] = df_plot['campaignID'].astype(str)

    # Dùng Plotly Express vẽ biểu đồ
    fig_bar_compare = px.bar(
        df_plot, 
        x='campaignID', 
        y=selected_plot_metric,
        color=selected_plot_metric,
        color_continuous_scale='Teal',
        text_auto='.2s',
        title=f'So sánh {selected_plot_metric} giữa Top 10 Campaign'
    )
    
    # CẤU HÌNH TRỤC X LÀ CATEGORY ĐỂ CÁC CỘT HIỂN THỊ TO, RÕ RÀNG
    fig_bar_compare.update_layout(
        xaxis_title="Campaign ID", 
        yaxis_title=selected_plot_metric,
        xaxis_type='category',             # <--- Chìa khóa xử lý lỗi nằm ở đây
        xaxis={'categoryorder': 'total descending'} # Sắp xếp cột từ cao xuống thấp cho đẹp
    )
    st.plotly_chart(fig_bar_compare, use_container_width=True)  
else:
    st.info("Vui lòng chọn ít nhất 1 chỉ số từ menu Dropdown để hiển thị dữ liệu.")

st.divider()

# # ==========================================
# # BƯỚC 4: HEATMAP DOANH THU THEO THÁNG
# # ==========================================
# st.write("### 📅 Biểu đồ nhiệt (Heatmap): Doanh thu theo tháng của Top 10")

# amount_col = "userChargeAmount" if "userChargeAmount" in df_transactions.columns else "amount"
# default_selection = top_10_ids[:5] if len(top_10_ids) >= 5 else top_10_ids

# selected_heatmap_campaigns = st.multiselect(
#     "Tùy chọn các Campaign (thuộc Top 10 Total Score) để hiển thị trên Heatmap:",
#     options=top_10_ids,
#     default=default_selection
# )

# if not selected_heatmap_campaigns:
#     st.info("Vui lòng chọn ít nhất 1 Campaign để hiển thị biểu đồ.")
# else:
#     df_heatmap_data = (
#         df_transactions
#         .filter(pl.col("campaignID").is_in(selected_heatmap_campaigns))
#         .with_columns([
#             pl.col("reqDate").cast(pl.String).str.slice(0, 7).alias("Tháng"), 
#             pl.col("campaignID").cast(pl.String) 
#         ])
#         .group_by(["campaignID", "Tháng"])
#         .agg(pl.col(amount_col).sum().alias("Doanh thu"))
#     ).to_pandas()
    
#     if not df_heatmap_data.empty:
#         df_pivot = df_heatmap_data.pivot(   
#             index="campaignID", 
#             columns="Tháng", 
#             values="Doanh thu"
#         ).fillna(0) 
        
#         fig_heatmap = px.imshow(
#             df_pivot,
#             text_auto=".2s", 
#             aspect="auto",
#             color_continuous_scale="Oranges", 
#             labels=dict(x="Thời gian (Tháng)", y="Campaign ID", color="Doanh thu (VNĐ)")
#         )
        
#         fig_heatmap.update_layout(
#             xaxis_title="Tháng",
#             yaxis_title="Campaign ID",
#             yaxis_type="category", 
#             yaxis={'categoryorder': 'total ascending'} 
#         )
        
#         st.plotly_chart(fig_heatmap, use_container_width=True)
#     else:
#         st.warning("Không có dữ liệu ngày tháng hợp lệ để vẽ Heatmap.")


