import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ──────────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    matplotlib.rcParams['font.family'] = 'NanumGothic'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 페이지 기본 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="이탈 고객 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 공통 CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 사이드바 */
    [data-testid="stSidebar"] {background-color: #1e2a3a;}
    [data-testid="stSidebar"] * {color: #ffffff !important;}

    /* 메인 타이틀 */
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.05rem; color: #6c757d; margin-bottom: 1.5rem;
    }

    /* 카드 */
    .card {
        background: #ffffff; border-radius: 12px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 4px solid #667eea;
    }
    .card-title {
        font-size: 1.05rem; font-weight: 700;
        color: #343a40; margin-bottom: 0.6rem;
    }

    /* 메트릭 박스 */
    .metric-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px; padding: 1rem 1.2rem;
        text-align: center; color: white;
    }
    .metric-box .val {font-size: 1.9rem; font-weight: 800;}
    .metric-box .lbl {font-size: 0.82rem; opacity: 0.85; margin-top: 2px;}

    /* 배지 */
    .badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
        background: #e9ecef; color: #495057; margin: 2px;
    }

    /* 구분선 */
    .section-divider {
        border: none; border-top: 2px solid #e9ecef; margin: 1.5rem 0;
    }

    /* 결과 테이블 */
    .result-table th {background: #667eea; color: white; text-align: center;}
    .result-table td {text-align: center;}

    /* 업로드 영역 */
    [data-testid="stFileUploader"] {
        border: 2px dashed #667eea !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }

    /* 버튼 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {opacity: 0.88;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  세션 상태 초기화
# ══════════════════════════════════════════════════════════════════
defaults = {
    "df_raw": None,          # 원본 데이터
    "df": None,              # 작업 데이터
    "df_processed": None,    # 전처리 완료 데이터
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "lr_model": None, "dt_model": None,
    "lr_result": None, "dt_result": None,
    "selected_X": [],
    "selected_y": None,
    "split_ratio": "7:3",
    "missing_handled": False,
    "outlier_handled": False,
    "encoded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
#  사이드바 네비게이션
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 이탈 고객 예측")
    st.markdown("---")
    pages = {
        "🏠  메인 / 데이터 업로드": "main",
        "🔍  데이터 탐색":          "eda",
        "⚙️  데이터 전처리":        "preprocess",
        "🤖  연구 모형":            "model",
        "📈  연구 결과":            "result",
    }
    page = st.radio("페이지 선택", list(pages.keys()), label_visibility="collapsed")
    current = pages[page]

    st.markdown("---")
    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(f"**📁 데이터 현황**")
        st.markdown(f"- 행: `{df_info.shape[0]:,}`")
        st.markdown(f"- 열: `{df_info.shape[1]}`")
        status_items = [
            ("결측치 처리", st.session_state.missing_handled),
            ("이상치 처리", st.session_state.outlier_handled),
            ("인코딩",     st.session_state.encoded),
        ]
        for label, done in status_items:
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {label}")
    else:
        st.markdown("*데이터를 먼저 업로드하세요*")

# ══════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ══════════════════════════════════════════════════════════════════
def check_data():
    if st.session_state.df is None:
        st.warning("⚠️ 먼저 **메인 페이지**에서 데이터를 업로드해 주세요.")
        st.stop()

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, "predict_proba") else None)
    fpr, tpr, _ = roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)
    roc_auc = auc(fpr, tpr) if fpr is not None else None
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "fpr": fpr, "tpr": tpr, "auc": roc_auc,
        "y_pred": y_pred,
        "cm": confusion_matrix(y_test, y_pred),
    }

# ══════════════════════════════════════════════════════════════════
#  PAGE 1 ── 메인 / 데이터 업로드
# ══════════════════════════════════════════════════════════════════
if current == "main":
    st.markdown('<p class="main-title">📊 신용평가모형</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">고객 이탈 예측을 위한 머신러닝 분석 플랫폼</p>',
                unsafe_allow_html=True)

    # 소개 카드
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🔍", "데이터 탐색",   "변수 분포·상관관계를 시각화합니다."),
        ("⚙️", "데이터 전처리", "결측치·이상치·인코딩을 처리합니다."),
        ("🤖", "예측 모형",     "Logistic Regression · Decision Tree"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <div class="card-title">{title}</div>
                <div style="color:#6c757d;font-size:0.88rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # 업로드 섹션
    st.markdown("### 📂 데이터 업로드")
    st.markdown("CSV 또는 Excel 파일을 업로드하세요.")

    uploaded = st.file_uploader(
        "파일 선택 (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        help="UTF-8 인코딩 CSV 또는 Excel 파일을 지원합니다."
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded, encoding="utf-8-sig")
            else:
                df = pd.read_excel(uploaded)

            # 세션 저장
            st.session_state.df_raw = df.copy()
            st.session_state.df = df.copy()
            st.session_state.df_processed = None
            # 전처리 상태 초기화
            for k in ["missing_handled", "outlier_handled", "encoded"]:
                st.session_state[k] = False
            for k in ["X_train", "X_test", "y_train", "y_test",
                      "lr_model", "dt_model", "lr_result", "dt_result"]:
                st.session_state[k] = None

            st.success(f"✅ **{uploaded.name}** 업로드 완료!")

            # 요약 메트릭
            c1, c2, c3, c4 = st.columns(4)
            metrics_data = [
                (df.shape[0], "총 행 수"),
                (df.shape[1], "총 열 수"),
                (int(df.isnull().sum().sum()), "결측치 수"),
                (df.select_dtypes(include=np.number).shape[1], "수치형 변수"),
            ]
            for col, (val, lbl) in zip([c1, c2, c3, c4], metrics_data):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="val">{val:,}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 미리보기
            with st.expander("📋 데이터 미리보기 (상위 10행)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            with st.expander("📊 기술 통계량"):
                st.dataframe(df.describe(include="all").T, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 파일 로드 오류: {e}")

    else:
        # 샘플 데이터 생성 버튼
        st.markdown("---")
        st.markdown("#### 💡 샘플 데이터로 시작하기")
        if st.button("🎲 샘플 데이터 생성"):
            np.random.seed(42)
            n = 500
            sample_df = pd.DataFrame({
                "age":           np.random.randint(20, 70, n),
                "tenure":        np.random.randint(1, 60, n),
                "balance":       np.random.uniform(0, 200000, n).round(2),
                "num_products":  np.random.randint(1, 5, n),
                "credit_score":  np.random.randint(300, 850, n),
                "is_active":     np.random.randint(0, 2, n),
                "gender":        np.random.choice(["Male", "Female"], n),
                "geography":     np.random.choice(["France", "Germany", "Spain"], n),
                "salary":        np.random.uniform(20000, 150000, n).round(2),
                "churn":         np.random.choice([0, 1], n, p=[0.8, 0.2]),
            })
            # 결측치 5% 삽입
            for col in ["balance", "credit_score", "salary"]:
                idx = np.random.choice(n, int(n * 0.05), replace=False)
                sample_df.loc[idx, col] = np.nan

            st.session_state.df_raw = sample_df.copy()
            st.session_state.df = sample_df.copy()
            for k in ["missing_handled", "outlier_handled", "encoded"]:
                st.session_state[k] = False
            st.success("✅ 샘플 데이터가 생성되었습니다! (500행 × 10열)")
            st.dataframe(sample_df.head(), use_container_width=True)
            st.rerun()

# ══════════════════════════════════════════════════════════════════
#  PAGE 2 ── 데이터 탐색 (EDA)
# ══════════════════════════════════════════════════════════════════
elif current == "eda":
    check_data()
    df = st.session_state.df

    st.markdown("## 🔍 데이터 탐색")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 기본 정보 ──────────────────────────────────────────────
    st.markdown("### 📐 기본 정보")
    c1, c2, c3, c4 = st.columns(4)
    info_data = [
        (df.shape[0], "행 수"),
        (df.shape[1], "열 수"),
        (int(df.isnull().sum().sum()), "결측치"),
        (df.duplicated().sum(), "중복 행"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], info_data):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{val:,}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 변수 목록 & 타입 ───────────────────────────────────────
    st.markdown("### 📋 변수 목록 및 타입")
    col_left, col_right = st.columns([1, 1])

    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    with col_left:
        dtype_df = pd.DataFrame({
            "변수명":         df.columns.tolist(),
            "데이터 타입":    df.dtypes.astype(str).values,
            "변수 구분":      ["수치형" if c in num_cols else "범주형" for c in df.columns],
            "결측치 수":      df.isnull().sum().values,
            "결측치 비율(%)": (df.isnull().mean() * 100).round(2).values,
            "고유값 수":      df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=320)

    with col_right:
        type_counts = pd.Series([
            "수치형" if pd.api.types.is_numeric_dtype(df[col]) else "범주형"
            for col in df.columns
        ]).value_counts()

        fig_pie, ax_pie = plt.subplots(figsize=(4, 3.5))
        colors = ["#667eea", "#f093fb"]
        ax_pie.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 11}
        )
        ax_pie.set_title("변수 타입 분포", fontsize=12, fontweight="bold")
        st.pyplot(fig_pie, use_container_width=True)
        plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 시각화 ─────────────────────────────────────────────────
    st.markdown("### 📊 변수 시각화")

    all_cols = df.columns.tolist()

    v_col1, v_col2, v_col3 = st.columns([1, 1, 1])
    with v_col1:
        x_var = st.selectbox("X축 변수", all_cols, key="eda_x")
    with v_col2:
        y_var = st.selectbox("Y축 변수", ["(없음)"] + all_cols, key="eda_y")
    with v_col3:
        chart_type = st.selectbox(
            "그래프 유형",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"],
            key="eda_chart"
        )

    if st.button("📊 그래프 생성", key="btn_chart"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        palette = "#667eea"

        try:
            x_data = df[x_var].copy()
            y_data = df[y_var].copy() if y_var != "(없음)" else None

            x_is_num = pd.api.types.is_numeric_dtype(x_data)
            y_is_num = pd.api.types.is_numeric_dtype(y_data) if y_data is not None else False

            # ── Histogram ─────────────────────────────────────
            if chart_type == "Histogram":
                if x_is_num:
                    ax.hist(
                        x_data.dropna().astype(float),
                        bins=30, color=palette,
                        edgecolor="white", alpha=0.85
                    )
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("빈도")
                else:
                    counts = x_data.value_counts()
                    ax.bar(
                        range(len(counts)), counts.values,
                        color=palette, edgecolor="white", alpha=0.85
                    )
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha="right")
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("빈도")

            # ── Box Plot ──────────────────────────────────────
            elif chart_type == "Box Plot":
                if not x_is_num:
                    st.warning("Box Plot의 X축은 수치형 변수를 선택해 주세요.")
                    plt.close()
                    st.stop()

                if y_data is not None and not y_is_num:
                    groups = []
                    labels = []
                    for g in df[y_var].dropna().unique():
                        grp = df[df[y_var] == g][x_var].dropna().astype(float)
                        if len(grp) > 0:
                            groups.append(grp.values)
                            labels.append(str(g))
                    ax.boxplot(
                        groups, patch_artist=True,
                        boxprops=dict(facecolor=palette, alpha=0.6),
                        medianprops=dict(color="red", linewidth=2)
                    )
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.set_xlabel(y_var)
                    ax.set_ylabel(x_var)
                else:
                    ax.boxplot(
                        x_data.dropna().astype(float).values,
                        patch_artist=True,
                        boxprops=dict(facecolor=palette, alpha=0.6),
                        medianprops=dict(color="red", linewidth=2)
                    )
                    ax.set_ylabel(x_var)
                    ax.set_xticks([1])
                    ax.set_xticklabels([x_var])

            # ── Scatter Plot ──────────────────────────────────
            elif chart_type == "Scatter Plot":
                if y_var == "(없음)":
                    st.warning("Scatter Plot은 Y축 변수를 선택해야 합니다.")
                    plt.close()
                    st.stop()
                if not x_is_num or not y_is_num:
                    st.warning("Scatter Plot은 X축, Y축 모두 수치형 변수를 선택해 주세요.")
                    plt.close()
                    st.stop()

                valid = df[[x_var, y_var]].dropna()
                ax.scatter(
                    valid[x_var].astype(float),
                    valid[y_var].astype(float),
                    alpha=0.4, color=palette,
                    edgecolors="white", linewidths=0.3
                )
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)

            # ── Bar Chart ─────────────────────────────────────
            elif chart_type == "Bar Chart":
                if not x_is_num:
                    counts = x_data.value_counts()
                    ax.bar(
                        range(len(counts)), counts.values,
                        color=palette, edgecolor="white", alpha=0.85
                    )
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha="right")
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("빈도")
                else:
                    if y_data is not None and y_is_num:
                        valid = df[[x_var, y_var]].dropna()
                        valid = valid.copy()
                        valid[x_var] = valid[x_var].astype(float)
                        valid[y_var] = valid[y_var].astype(float)
                        bins = pd.cut(valid[x_var], bins=10)
                        grouped = valid.groupby(bins, observed=True)[y_var].mean()
                        labels = [str(i) for i in grouped.index]
                        ax.bar(
                            range(len(grouped)), grouped.values,
                            color=palette, edgecolor="white", alpha=0.85
                        )
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                        ax.set_xlabel(x_var)
                        ax.set_ylabel(f"{y_var} (평균)")
                    else:
                        counts = x_data.value_counts().sort_index()
                        ax.bar(
                            range(len(counts)), counts.values,
                            color=palette, edgecolor="white", alpha=0.85
                        )
                        ax.set_xticks(range(len(counts)))
                        ax.set_xticklabels(
                            [str(i) for i in counts.index],
                            rotation=45, ha="right", fontsize=8
                        )
                        ax.set_xlabel(x_var)
                        ax.set_ylabel("빈도")

            # ── Line Chart ────────────────────────────────────
            elif chart_type == "Line Chart":
                if not x_is_num:
                    st.warning("Line Chart의 X축은 수치형 변수를 선택해 주세요.")
                    plt.close()
                    st.stop()

                if y_data is not None and y_is_num:
                    valid = df[[x_var, y_var]].dropna().sort_values(x_var)
                    ax.plot(
                        valid[x_var].astype(float),
                        valid[y_var].astype(float),
                        color=palette, alpha=0.8, linewidth=1.5
                    )
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                else:
                    sorted_data = x_data.dropna().astype(float).reset_index(drop=True)
                    ax.plot(
                        sorted_data.index,
                        sorted_data.values,
                        color=palette, alpha=0.8, linewidth=1.5
                    )
                    ax.set_xlabel("Index")
                    ax.set_ylabel(x_var)

            # ── 공통 스타일 ───────────────────────────────────
            title = f"{chart_type}  |  {x_var}"
            if y_var != "(없음)":
                title += f"  vs  {y_var}"
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"그래프 생성 오류: {e}")
        finally:
            plt.close()

    # ── 상관관계 히트맵 ────────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🌡️ 수치형 변수 상관관계 히트맵")
        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0, ax=ax_hm,
            linewidths=0.5, cbar_kws={"shrink": 0.8}
        )
        ax_hm.set_title("상관관계 히트맵", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_hm, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 ── 데이터 전처리 / Feature Selection / Partitioning
# ══════════════════════════════════════════════════════════════════
elif current == "preprocess":
    check_data()
    df = st.session_state.df.copy()

    st.markdown("## ⚙️ 데이터 전처리")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### 🧹 데이터 전처리")

    tab1, tab2, tab3, tab4 = st.tabs(["결측치 처리", "이상치 처리", "수치형 변환", "원핫 인코딩"])

    # ── 결측치 ──────────────────────────────────────────────
    with tab1:
        missing = st.session_state.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("✅ 결측치가 없습니다.")
        else:
            miss_df = pd.DataFrame({
                "변수명": missing.index,
                "결측치 수": missing.values,
                "결측치 비율(%)": (missing / len(st.session_state.df) * 100).round(2).values
            })
            st.dataframe(miss_df, use_container_width=True)

        method = st.radio(
            "처리 방법",
            ["평균값 대체 (수치형)", "중앙값 대체 (수치형)",
             "최빈값 대체 (범주형)", "행 삭제"],
            horizontal=True,
            key="missing_method"
        )

        if st.button("✅ 결측치 처리 실행", key="btn_missing"):
            df_work = st.session_state.df.copy()
            num_cols = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
            cat_cols = [c for c in df_work.columns if not pd.api.types.is_numeric_dtype(df_work[c])]

            if method == "행 삭제":
                df_work = df_work.dropna().reset_index(drop=True)

            else:
                # ✅ 수치형 처리
                for c in num_cols:
                    if df_work[c].isnull().any():
                        if method == "평균값 대체 (수치형)":
                            df_work[c] = df_work[c].fillna(df_work[c].mean())
                        else:  # 중앙값 or 최빈값 선택 시에도 수치형은 중앙값으로
                            df_work[c] = df_work[c].fillna(df_work[c].median())

                # ✅ 범주형 처리 (항상 최빈값으로)
                for c in cat_cols:
                    if df_work[c].isnull().any():
                        mode_val = df_work[c].mode()
                        if len(mode_val) > 0:
                            df_work[c] = df_work[c].fillna(mode_val[0])

            # ✅ 그래도 남은 NaN 강제 제거 (최종 안전망)
            remaining = df_work.isnull().sum().sum()
            if remaining > 0:
                for c in df_work.columns:
                    if df_work[c].isnull().any():
                        if pd.api.types.is_numeric_dtype(df_work[c]):
                            df_work[c] = df_work[c].fillna(df_work[c].median())
                        else:
                            df_work[c] = df_work[c].fillna("Unknown")

            st.session_state.df = df_work
            st.session_state.missing_handled = True

            final_nan = df_work.isnull().sum().sum()
            if final_nan == 0:
                st.success(f"✅ 결측치 처리 완료! 남은 결측치: 0개 🎉")
            else:
                st.warning(f"⚠️ 처리 후 남은 결측치: {final_nan}개")
            st.rerun()

    # ── 이상치 ──────────────────────────────────────────────
    with tab2:
        df_cur = st.session_state.df

        num_cols = []
        for c in df_cur.columns:
            if pd.api.types.is_numeric_dtype(df_cur[c]):
                try:
                    test = df_cur[c].dropna().astype(float).values
                    _ = np.quantile(test, 0.25)
                    num_cols.append(c)
                except Exception:
                    pass

        if not num_cols:
            st.info("수치형 변수가 없습니다.")
        else:
            outlier_info = []
            for c in num_cols:
                try:
                    arr = df_cur[c].dropna().astype(float).values
                    Q1 = float(np.quantile(arr, 0.25))
                    Q3 = float(np.quantile(arr, 0.75))
                    IQR = Q3 - Q1
                    n_out = int(((arr < Q1 - 1.5 * IQR) | (arr > Q3 + 1.5 * IQR)).sum())
                    outlier_info.append({
                        "변수명": c,
                        "이상치 수": n_out,
                        "Q1": round(Q1, 2),
                        "Q3": round(Q3, 2),
                        "IQR": round(IQR, 2)
                    })
                except Exception:
                    outlier_info.append({
                        "변수명": c,
                        "이상치 수": "계산불가",
                        "Q1": "-", "Q3": "-", "IQR": "-"
                    })

            st.dataframe(pd.DataFrame(outlier_info), use_container_width=True)

            out_method = st.radio(
                "처리 방법",
                ["IQR 기반 클리핑 (Winsorizing)", "IQR 기반 행 제거"],
                horizontal=True,
                key="outlier_method"
            )
            out_cols = st.multiselect(
                "처리할 변수 선택",
                num_cols,
                default=num_cols,
                key="outlier_cols"
            )

            if st.button("✅ 이상치 처리 실행", key="btn_outlier"):
                df_work = st.session_state.df.copy()
                processed = []
                skipped = []

                for c in out_cols:
                    try:
                        arr = df_work[c].dropna().astype(float).values
                        Q1 = float(np.quantile(arr, 0.25))
                        Q3 = float(np.quantile(arr, 0.75))
                        IQR = Q3 - Q1
                        lo = Q1 - 1.5 * IQR
                        hi = Q3 + 1.5 * IQR

                        df_work[c] = df_work[c].astype(float)

                        if out_method == "IQR 기반 클리핑 (Winsorizing)":
                            df_work[c] = df_work[c].clip(lo, hi)
                        else:
                            df_work = df_work[
                                (df_work[c] >= lo) & (df_work[c] <= hi)
                            ]
                        processed.append(c)
                    except Exception as e:
                        skipped.append(f"{c} ({e})")

                # ✅ 행 제거 후 인덱스 리셋
                df_work = df_work.reset_index(drop=True)

                st.session_state.df = df_work
                st.session_state.outlier_handled = True

                if processed:
                    st.success(
                        f"✅ 처리 완료! 변수: {processed} / 현재 행 수: {len(df_work):,}"
                    )
                if skipped:
                    st.warning(f"⚠️ 처리 실패: {skipped}")
                st.rerun()

    # ── 수치형 변환 ────────────────────────────────────
    with tab3:
        df_cur = st.session_state.df
        cat_cols_all = [c for c in df_cur.columns if not pd.api.types.is_numeric_dtype(df_cur[c])]

        st.markdown("#### 🔢 문자열 → 수치형 변환")
        st.markdown("""
        `%`, `₩`, `,` 등 특수문자가 포함된 숫자형 문자열을 수치형으로 변환합니다.  
        예) `'13.5%'` → `13.5` / `'1,000'` → `1000`
        """)

        if not cat_cols_all:
            st.success("✅ 변환할 범주형 변수가 없습니다.")
        else:
            # 자동 감지
            auto_detect = []
            for c in cat_cols_all:
                sample = df_cur[c].dropna().astype(str).head(50)
                cleaned = sample.str.replace(r'[%,₩$\s]', '', regex=True)
                try:
                    pd.to_numeric(cleaned)
                    auto_detect.append(c)
                except Exception:
                    numeric_ratio = pd.to_numeric(cleaned, errors='coerce').notna().mean()
                    if numeric_ratio > 0.8:
                        auto_detect.append(c)

            if auto_detect:
                st.info(f"💡 수치형으로 변환 가능한 변수가 감지되었습니다: **{auto_detect}**")

            convert_cols = st.multiselect(
                "수치형으로 변환할 변수 선택",
                cat_cols_all,
                default=auto_detect,
                key="convert_cols"
            )

            remove_chars = st.text_input(
                "제거할 특수문자 (기본: % , ₩ $ 공백)",
                value="%,₩$ ",
                key="remove_chars"
            )

            col_prev1, col_prev2 = st.columns(2)
            with col_prev1:
                st.markdown("**변환 전 미리보기**")
                if convert_cols:
                    st.dataframe(
                        df_cur[convert_cols].head(5).astype(str),
                        use_container_width=True
                    )

            with col_prev2:
                st.markdown("**변환 후 미리보기**")
                if convert_cols:
                    preview = df_cur[convert_cols].head(5).copy()
                    for c in convert_cols:
                        pattern = f"[{re.escape(remove_chars)}]"
                        preview[c] = pd.to_numeric(
                            preview[c].astype(str).str.replace(pattern, '', regex=True),
                            errors='coerce'
                        )
                    st.dataframe(preview, use_container_width=True)

            if st.button("✅ 수치형 변환 실행", key="btn_convert"):
                if not convert_cols:
                    st.warning("변환할 변수를 선택해 주세요.")
                else:
                    df_work = st.session_state.df.copy()
                    success_cols = []
                    fail_cols = []

                    for c in convert_cols:
                        try:
                            pattern = f"[{re.escape(remove_chars)}]"
                            converted = pd.to_numeric(
                                df_work[c].astype(str).str.replace(pattern, '', regex=True),
                                errors='coerce'
                            )
                            success_rate = converted.notna().mean()

                            if success_rate >= 0.8:
                                # ✅ 변환 후 남은 NaN → 중앙값으로 대체
                                if converted.isnull().any():
                                    median_val = converted.median()
                                    converted = converted.fillna(median_val)

                                df_work[c] = converted
                                success_cols.append(f"{c} ({success_rate:.0%})")
                            else:
                                fail_cols.append(f"{c} ({success_rate:.0%})")
                        except Exception as e:
                            fail_cols.append(f"{c} (오류: {e})")

                    st.session_state.df = df_work
                    st.session_state.convert_handled = True  # ✅ 완료 플래그 저장

                    if success_cols:
                        st.success(f"✅ 변환 완료: {success_cols}")
                    if fail_cols:
                        st.warning(f"⚠️ 변환 실패 (수치 비율 80% 미만): {fail_cols}")

                    # ✅ 메시지 확인 후 rerun (0.5초 딜레이)
                    import time
                    time.sleep(0.5)
                    st.rerun()

    # ── 원핫 인코딩 ─────────────────────────────────────────
    with tab4:
        df_cur = st.session_state.df
        cat_cols = [c for c in df_cur.columns if not pd.api.types.is_numeric_dtype(df_cur[c])]

        if not cat_cols:
            st.success("✅ 인코딩할 범주형 변수가 없습니다.")
        else:
            st.markdown("**범주형 변수 목록**")
            cat_info = pd.DataFrame({
                "변수명":    cat_cols,
                "고유값 수": [df_cur[c].nunique() for c in cat_cols],
                "샘플값":    [str(df_cur[c].dropna().unique()[:3].tolist()) for c in cat_cols],
            })
            st.dataframe(cat_info, use_container_width=True)

            high_cardinality = [c for c in cat_cols if df_cur[c].nunique() > 10]
            if high_cardinality:
                st.warning(f"""
                ⚠️ **고유값이 10개 초과인 변수** (One-Hot 인코딩 시 컬럼 폭발 주의):  
                {high_cardinality}  
                → **수치형 변환 탭**에서 먼저 처리하거나, Label Encoding을 권장합니다.
                """)

            safe_cols = [c for c in cat_cols if df_cur[c].nunique() <= 10]

            enc_method = st.radio(
                "인코딩 방법",
                ["One-Hot Encoding", "Label Encoding"],
                horizontal=True,
                key="enc_method"
            )

            if enc_method == "One-Hot Encoding":
                enc_cols = st.multiselect(
                    "인코딩할 변수 선택",
                    cat_cols,
                    default=safe_cols,
                    key="enc_cols_ohe",
                    help="고유값이 적은 변수만 선택을 권장합니다."
                )
                if enc_cols:
                    expected_new_cols = sum(df_cur[c].nunique() - 1 for c in enc_cols)
                    remaining_cols = len([c for c in df_cur.columns if c not in enc_cols])
                    total_expected = remaining_cols + expected_new_cols
                    current_cols = df_cur.shape[1]

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("현재 컬럼 수", f"{current_cols}개")
                    with col_b:
                        st.metric("인코딩 후 예상 컬럼 수", f"{total_expected}개")
                    with col_c:
                        delta = total_expected - current_cols
                        st.metric("증가량", f"+{delta}개",
                                  delta_color="inverse" if delta > 50 else "normal")
            else:
                enc_cols = st.multiselect(
                    "인코딩할 변수 선택",
                    cat_cols,
                    default=cat_cols,
                    key="enc_cols_le"
                )

            if st.button("✅ 인코딩 실행", key="btn_encode"):
                if not enc_cols:
                    st.warning("인코딩할 변수를 선택해 주세요.")
                else:
                    df_work = st.session_state.df.copy()
                    try:
                        if enc_method == "One-Hot Encoding":
                            df_work = pd.get_dummies(df_work, columns=enc_cols, drop_first=True)

                            # ✅ bool 타입 → int로 변환 (pandas 2.0+ 대응)
                            bool_cols = [c for c in df_work.columns if df_work[c].dtype == bool]
                            if bool_cols:
                                df_work[bool_cols] = df_work[bool_cols].astype(int)

                        else:  # Label Encoding
                            le = LabelEncoder()
                            for c in enc_cols:
                                # ✅ NaN 먼저 처리 후 인코딩
                                if df_work[c].isnull().any():
                                    mode_val = df_work[c].mode()
                                    df_work[c] = df_work[c].fillna(
                                        mode_val[0] if len(mode_val) > 0 else "Unknown"
                                    )
                                df_work[c] = le.fit_transform(df_work[c].astype(str))

                        # ✅ 인코딩 후 남은 object 컬럼 확인 및 경고
                        remaining_obj = [
                            c for c in df_work.columns
                            if df_work[c].dtype == object and c not in enc_cols
                        ]
                        if remaining_obj:
                            st.warning(f"⚠️ 아직 인코딩되지 않은 범주형 변수: {remaining_obj}")

                        # ✅ 최종 NaN 확인
                        final_nan = df_work.isnull().sum().sum()
                        if final_nan > 0:
                            st.warning(f"⚠️ 인코딩 후 NaN {final_nan}개 감지 → 자동 제거")
                            for c in df_work.columns:
                                if df_work[c].isnull().any():
                                    if pd.api.types.is_numeric_dtype(df_work[c]):
                                        df_work[c] = df_work[c].fillna(df_work[c].median())
                                    else:
                                        df_work[c] = df_work[c].fillna("Unknown")

                        st.session_state.df = df_work
                        st.session_state.encoded = True
                        st.success(f"✅ 인코딩 완료! (현재 열 수: {df_work.shape[1]})")

                        import time
                        time.sleep(0.5)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 인코딩 오류: {e}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── SECTION 2: Feature Selection ──────────────────────────
    st.markdown("### 🎯 Feature Selection")

    df_cur = st.session_state.df
    all_cols = df_cur.columns.tolist()

    fs_col1, fs_col2 = st.columns(2)
    with fs_col1:
        selected_y = st.selectbox(
            "종속변수 Y (타겟)",
            all_cols,
            index=len(all_cols) - 1,
            help="예측하려는 목표 변수를 선택하세요.",
            key="sel_y"
        )
    with fs_col2:
        x_options = [c for c in all_cols if c != selected_y]
        selected_X = st.multiselect(
            "독립변수 X (피처)",
            x_options,
            default=x_options,
            help="모델 학습에 사용할 변수를 선택하세요.",
            key="sel_x"
        )

    if st.button("✅ Feature Selection 저장", key="btn_fs"):
        if not selected_X:
            st.error("❌ 독립변수를 1개 이상 선택해 주세요.")
        else:
            st.session_state.selected_X = selected_X
            st.session_state.selected_y = selected_y
            st.success(f"✅ 저장 완료! X: {len(selected_X)}개 변수 / Y: {selected_y}")

    if st.session_state.selected_X:
        with st.expander("현재 선택된 변수 확인"):
            st.markdown(f"**Y (종속변수):** `{st.session_state.selected_y}`")
            st.markdown("**X (독립변수):**")
            cols_display = st.columns(4)
            for i, c in enumerate(st.session_state.selected_X):
                cols_display[i % 4].markdown(
                    f'<span class="badge">{c}</span>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── SECTION 3: Data Partitioning ──────────────────────────
    st.markdown("### ✂️ Data Partitioning")

    dp_col1, dp_col2 = st.columns([1, 2])
    with dp_col1:
        split_ratio = st.radio(
            "Train : Test 비율",
            ["7:3", "8:2"],
            index=0,
            help="학습 데이터와 테스트 데이터의 비율을 선택하세요.",
            key="split_ratio_radio"
        )
        random_seed = st.number_input("Random Seed", value=42, min_value=0, key="random_seed")

    with dp_col2:
        ratio_val = 0.7 if split_ratio == "7:3" else 0.8
        n_total = len(df_cur)
        n_train = int(n_total * ratio_val)
        n_test  = n_total - n_train

        st.markdown(f"""
        <div class="card">
            <div class="card-title">📊 분할 미리보기</div>
            <div style="display:flex; gap:1rem; margin-top:0.5rem;">
                <div style="flex:1; background:#667eea; border-radius:8px;
                            padding:0.8rem; text-align:center; color:white;">
                    <div style="font-size:1.5rem; font-weight:800;">{n_train:,}</div>
                    <div style="font-size:0.82rem;">Train ({int(ratio_val*100)}%)</div>
                </div>
                <div style="flex:1; background:#764ba2; border-radius:8px;
                            padding:0.8rem; text-align:center; color:white;">
                    <div style="font-size:1.5rem; font-weight:800;">{n_test:,}</div>
                    <div style="font-size:0.82rem;">Test ({int((1-ratio_val)*100)}%)</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    if st.button("✅ 데이터 분할 실행", key="btn_split"):
        if not st.session_state.selected_X or not st.session_state.selected_y:
            st.error("❌ Feature Selection을 먼저 완료해 주세요.")
        else:
            try:
                X = df_cur[st.session_state.selected_X]
                y = df_cur[st.session_state.selected_y]

                non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
                if non_num:
                    st.warning(f"⚠️ 비수치형 변수 {non_num}는 제외됩니다. 인코딩 후 다시 시도하세요.")
                    X = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]]

                test_size = 1 - ratio_val
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=int(random_seed),
                    stratify=y
                )
                st.session_state.X_train = X_tr
                st.session_state.X_test  = X_te
                st.session_state.y_train = y_tr
                st.session_state.y_test  = y_te
                st.session_state.split_ratio = split_ratio

                st.success(
                    f"✅ 분할 완료! Train: {len(X_tr):,}행 / Test: {len(X_te):,}행"
                )
            except Exception as e:
                st.error(f"❌ 분할 오류: {e}")


# ══════════════════════════════════════════════════════════════════
#  PAGE 4 ── 연구 모형
# ══════════════════════════════════════════════════════════════════
elif current == "model":
    check_data()
    st.markdown("## 🤖 연구 모형")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.session_state.X_train is None:
        st.warning("⚠️ **데이터 전처리** 페이지에서 데이터 분할을 먼저 완료해 주세요.")
        st.stop()

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test

    # 데이터 현황
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📊 학습 데이터 현황</div>
        Train: <b>{len(X_train):,}행</b> &nbsp;|&nbsp;
        Test: <b>{len(X_test):,}행</b> &nbsp;|&nbsp;
        Feature 수: <b>{X_train.shape[1]}</b> &nbsp;|&nbsp;
        Target: <b>{st.session_state.selected_y}</b>
    </div>""", unsafe_allow_html=True)

    model_tab1, model_tab2 = st.tabs(
        ["📉 Logistic Regression", "🌳 Decision Tree"]
    )

    # ── Logistic Regression ─────────────────────────────────
    with model_tab1:
        st.markdown("#### Logistic Regression 하이퍼파라미터")
        lr_col1, lr_col2, lr_col3 = st.columns(3)
        with lr_col1:
            lr_C = st.select_slider("규제 강도 C",
                                    options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                                    value=1.0)
        with lr_col2:
            lr_max_iter = st.slider("최대 반복 횟수", 100, 2000, 1000, 100)
        with lr_col3:
            lr_solver = st.selectbox("Solver",
                                     ["lbfgs", "liblinear", "saga", "newton-cg"])

        if st.button("🚀 Logistic Regression 학습", key="btn_lr"):
            with st.spinner("학습 중..."):
                try:
                    lr = LogisticRegression(
                        C=lr_C, max_iter=lr_max_iter,
                        solver=lr_solver, random_state=42
                    )
                    lr.fit(X_train, y_train)
                    st.session_state.lr_model  = lr
                    st.session_state.lr_result = compute_metrics(lr, X_test, y_test)
                    st.success("✅ Logistic Regression 학습 완료!")
                except Exception as e:
                    st.error(f"❌ 학습 오류: {e}")

        if st.session_state.lr_result:
            r = st.session_state.lr_result
            st.markdown("##### 📊 학습 결과")
            m1, m2, m3, m4 = st.columns(4)
            for col, (lbl, val) in zip(
                [m1, m2, m3, m4],
                [("Accuracy", r["accuracy"]), ("Precision", r["precision"]),
                 ("Recall",   r["recall"]),   ("F1-Score",  r["f1"])]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="val">{val:.4f}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # 혼동 행렬
            st.markdown("<br>", unsafe_allow_html=True)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                        ax=ax_cm, linewidths=0.5)
            ax_cm.set_title("Confusion Matrix", fontweight="bold")
            ax_cm.set_xlabel("예측값"); ax_cm.set_ylabel("실제값")
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=False)
            plt.close()

    # ── Decision Tree ───────────────────────────────────────
    with model_tab2:
        st.markdown("#### Decision Tree 하이퍼파라미터")
        dt_col1, dt_col2, dt_col3 = st.columns(3)
        with dt_col1:
            dt_max_depth = st.slider("최대 깊이 (max_depth)", 1, 20, 5)
        with dt_col2:
            dt_min_samples = st.slider("최소 분할 샘플 수", 2, 50, 2)
        with dt_col3:
            dt_criterion = st.selectbox("분할 기준", ["gini", "entropy", "log_loss"])

        if st.button("🚀 Decision Tree 학습", key="btn_dt"):
            with st.spinner("학습 중..."):
                try:
                    dt = DecisionTreeClassifier(
                        max_depth=dt_max_depth,
                        min_samples_split=dt_min_samples,
                        criterion=dt_criterion,
                        random_state=42
                    )
                    dt.fit(X_train, y_train)
                    st.session_state.dt_model  = dt
                    st.session_state.dt_result = compute_metrics(dt, X_test, y_test)
                    st.success("✅ Decision Tree 학습 완료!")
                except Exception as e:
                    st.error(f"❌ 학습 오류: {e}")

        if st.session_state.dt_result:
            r = st.session_state.dt_result
            st.markdown("##### 📊 학습 결과")
            m1, m2, m3, m4 = st.columns(4)
            for col, (lbl, val) in zip(
                [m1, m2, m3, m4],
                [("Accuracy", r["accuracy"]), ("Precision", r["precision"]),
                 ("Recall",   r["recall"]),   ("F1-Score",  r["f1"])]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="val">{val:.4f}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # Feature Importance
            st.markdown("<br>", unsafe_allow_html=True)
            fi = pd.Series(
                st.session_state.dt_model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=True).tail(15)

            fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
            fi.plot(kind="barh", ax=ax_fi, color="#667eea", edgecolor="white")
            ax_fi.set_title("Feature Importance (Top 15)", fontweight="bold")
            ax_fi.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()

# ══════════════════════════════════════════════════════════════════
#  PAGE 5 ── 연구 결과
# ══════════════════════════════════════════════════════════════════
elif current == "result":
    check_data()
    st.markdown("## 📈 연구 결과")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    lr_r = st.session_state.lr_result
    dt_r = st.session_state.dt_result

    if lr_r is None and dt_r is None:
        st.warning("⚠️ **연구 모형** 페이지에서 모델을 먼저 학습해 주세요.")
        st.stop()

    # ── 성능 비교 테이블 ───────────────────────────────────────
    st.markdown("### 📊 모형 성능 비교")

    metrics_rows = []
    for name, r in [("Logistic Regression", lr_r), ("Decision Tree", dt_r)]:
        if r:
            metrics_rows.append({
                "모형":       name,
                "Accuracy":  f"{r['accuracy']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall":    f"{r['recall']:.4f}",
                "F1-Score":  f"{r['f1']:.4f}",
                "AUC":       f"{r['auc']:.4f}" if r["auc"] else "N/A",
            })

    if metrics_rows:
        result_df = pd.DataFrame(metrics_rows).set_index("모형")
        st.dataframe(result_df.style.highlight_max(
            axis=0, color="#d4edda", subset=["Accuracy", "Precision",
                                              "Recall", "F1-Score", "AUC"]
        ), use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 메트릭 시각화 ─────────────────────────────────────────
    st.markdown("### 📉 성능 지표 시각화")

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_vals = ([lr_r["accuracy"], lr_r["precision"], lr_r["recall"], lr_r["f1"]]
               if lr_r else None)
    dt_vals = ([dt_r["accuracy"], dt_r["precision"], dt_r["recall"], dt_r["f1"]]
               if dt_r else None)

    fig_bar, ax_bar = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metric_names))
    width = 0.35

    if lr_vals:
        bars1 = ax_bar.bar(x - width/2, lr_vals, width,
                           label="Logistic Regression",
                           color="#667eea", edgecolor="white", alpha=0.9)
        for bar in bars1:
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    if dt_vals:
        bars2 = ax_bar.bar(x + width/2, dt_vals, width,
                           label="Decision Tree",
                           color="#764ba2", edgecolor="white", alpha=0.9)
        for bar in bars2:
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=11)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("모형별 성능 지표 비교", fontsize=13, fontweight="bold")
    ax_bar.legend(fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── ROC Curve ─────────────────────────────────────────────
    st.markdown("### 📈 ROC Curve 비교")

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random (AUC = 0.50)")

    colors = {"Logistic Regression": "#667eea", "Decision Tree": "#764ba2"}
    for name, r in [("Logistic Regression", lr_r), ("Decision Tree", dt_r)]:
        if r and r["fpr"] is not None:
            ax_roc.plot(r["fpr"], r["tpr"],
                        color=colors[name], lw=2.5,
                        label=f"{name} (AUC = {r['auc']:.4f})")
            ax_roc.fill_between(r["fpr"], r["tpr"], alpha=0.08,
                                color=colors[name])

    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax_roc.set_title("ROC Curve 비교", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=11)
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_roc, use_container_width=True)
    plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 혼동 행렬 비교 ─────────────────────────────────────────
    st.markdown("### 🔲 Confusion Matrix 비교")
    cm_cols = st.columns(2)
    for col, (name, r) in zip(
        cm_cols,
        [("Logistic Regression", lr_r), ("Decision Tree", dt_r)]
    ):
        if r:
            with col:
                st.markdown(f"**{name}**")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(r["cm"], annot=True, fmt="d",
                            cmap="Blues", ax=ax_cm, linewidths=0.5,
                            cbar_kws={"shrink": 0.8})
                ax_cm.set_title(f"{name}", fontsize=11, fontweight="bold")
                ax_cm.set_xlabel("예측값"); ax_cm.set_ylabel("실제값")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 결과 다운로드 ─────────────────────────────────────────
    st.markdown("### 💾 결과 저장")
    if metrics_rows:
        csv_result = result_df.to_csv(encoding="utf-8-sig")
        st.download_button(
            label="📥 성능 지표 CSV 다운로드",
            data=csv_result,
            file_name="model_performance.csv",
            mime="text/csv"
        )
