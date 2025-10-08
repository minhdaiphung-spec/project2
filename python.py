import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
import json
import math

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh",
    layout="wide"
)

st.title("Ứng Dụng Đánh Giá Phương Án Kinh Doanh 💰")
st.markdown("Sử dụng AI để trích xuất thông tin, xây dựng dòng tiền và đánh giá hiệu quả dự án.")

# --- ĐỊNH NGHĨA SCHEMA JSON CHO AI TRÍCH XUẤT DỮ LIỆU ---
FINANCIAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "initial_investment": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu (Vốn đầu tư) tại năm T=0. Đơn vị: Triệu VNĐ."},
        "project_lifespan": {"type": "INTEGER", "description": "Dòng đời dự án theo năm (Năm)."},
        "annual_revenue": {"type": "NUMBER", "description": "Doanh thu hàng năm dự kiến (Giả định không đổi). Đơn vị: Triệu VNĐ."},
        "annual_cost": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm dự kiến (Chưa bao gồm thuế, giả định không đổi). Đơn vị: Triệu VNĐ."},
        "wacc": {"type": "NUMBER", "description": "Chi phí sử dụng vốn bình quân (WACC) hay tỷ lệ chiết khấu, dưới dạng thập phân (ví dụ: 0.12 cho 12%)."},
        "tax_rate": {"type": "NUMBER", "description": "Thuế suất thu nhập doanh nghiệp, dưới dạng thập phân (ví dụ: 0.2 cho 20%)."}
    },
    "required": ["initial_investment", "project_lifespan", "annual_revenue", "annual_cost", "wacc", "tax_rate"]
}

# --- KHỞI TẠO STATE ---
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "cash_flow_df" not in st.session_state:
    st.session_state.cash_flow_df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

# --- HÀM 1: LỌC DỮ LIỆU BẰNG AI (TRẢ VỀ JSON CÓ CẤU TRÚC) ---
def extract_data_with_ai(text_content, api_key):
    """Gửi nội dung văn bản tới Gemini để trích xuất dữ liệu tài chính theo JSON Schema."""
    try:
        client = genai.Client(api_key=api_key)
        
        system_prompt = f"""Bạn là một chuyên gia trích xuất dữ liệu tài chính. Hãy đọc kỹ nội dung phương án kinh doanh sau và trích xuất 6 thông số sau: Vốn đầu tư ban đầu (Initial Investment), Dòng đời dự án (Project Lifespan), Doanh thu hàng năm (Annual Revenue), Chi phí hoạt động hàng năm (Annual Cost), WACC (WACC), và Thuế suất (Tax Rate).
        Đảm bảo đầu ra của bạn là một đối tượng JSON hoàn chỉnh, tuân thủ chính xác Schema đã cho. Đơn vị tiền tệ là Triệu VNĐ và các tỷ lệ (WACC, Thuế) phải ở dạng thập phân.
        Nội dung Phương án Kinh doanh: {text_content}"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FINANCIAL_SCHEMA,
                system_instruction=system_prompt
            )
        )
        
        # Parse JSON output
        json_string = response.text.strip().strip('`')
        if json_string.startswith('json'):
            json_string = json_string[4:].strip()
        
        return json.loads(json_string)

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi giải mã JSON từ AI. Vui lòng kiểm tra nội dung file có đủ thông tin không.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None

# --- HÀM 2: XÂY DỰNG BẢNG DÒNG TIỀN VÀ TÍNH CHỈ SỐ ---
@st.cache_data
def calculate_metrics(data):
    """Xây dựng dòng tiền dự án và tính toán NPV, IRR, PP, DPP."""
    
    # 1. Trích xuất thông số
    I0 = data['initial_investment'] # Vốn đầu tư (Triệu VNĐ)
    N = data['project_lifespan'] # Dòng đời (Năm)
    Rev = data['annual_revenue'] # Doanh thu (Triệu VNĐ)
    Cost = data['annual_cost'] # Chi phí (Triệu VNĐ)
    WACC = data['wacc'] # Tỷ lệ chiết khấu
    Tax = data['tax_rate'] # Thuế suất

    # 2. Xây dựng Dòng tiền thuần (Net Cash Flow - NCF)
    
    # Giả định: NCF = (Rev - Cost) * (1 - Tax) (Bỏ qua Khấu hao, Vốn luân chuyển, Giá trị thanh lý)
    EBIT = Rev - Cost
    EAT = EBIT * (1 - Tax)
    NCF_annual = EAT # Vì không có Khấu hao/Thay đổi vốn luân chuyển
    
    # Tạo mảng năm và dòng tiền
    years = list(range(0, N + 1)) # Năm 0 đến Năm N
    
    # Dòng tiền khởi đầu (năm 0)
    cash_flows = [-I0] 
    
    # Dòng tiền hàng năm (năm 1 đến năm N)
    for _ in range(N):
        cash_flows.append(NCF_annual)
        
    # Tạo DataFrame dòng tiền
    df = pd.DataFrame({
        'Năm': years,
        'Dòng tiền thuần (NCF)': cash_flows,
    })

    # 3. Tính toán các chỉ số
    
    # a. NPV (Net Present Value)
    npv_value = np.npv(WACC, cash_flows)

    # b. IRR (Internal Rate of Return)
    try:
        irr_value = np.irr(cash_flows)
    except ValueError:
        irr_value = np.nan # Không thể tính nếu dòng tiền không đổi dấu

    # c. PP (Payback Period - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(cash_flows)
    pp = 0
    for i in range(1, N + 1):
        if cumulative_cf[i] >= 0:
            # Hoàn vốn trong năm i. Tính thời gian chính xác
            pp = i - 1 + abs(cumulative_cf[i-1]) / cash_flows[i]
            break
        elif i == N and cumulative_cf[i] < 0:
            pp = float('inf') # Dự án không hoàn vốn
            
    # d. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    discounted_cf = [cf / (1 + WACC)**t for t, cf in enumerate(cash_flows)]
    cumulative_discounted_cf = np.cumsum(discounted_cf)
    dpp = 0
    for i in range(1, N + 1):
        if cumulative_discounted_cf[i] >= 0:
            dpp = i - 1 + abs(cumulative_discounted_cf[i-1]) / discounted_cf[i]
            break
        elif i == N and cumulative_discounted_cf[i] < 0:
            dpp = float('inf') # Dự án không hoàn vốn chiết khấu
            
    # 4. Cập nhật DataFrame với các cột tính toán
    df['Chiết khấu (1/(1+WACC)^t)'] = [1 / (1 + WACC)**t for t in years]
    df['Dòng tiền chiết khấu (DCF)'] = [cf * df['Chiết khấu (1/(1+WACC)^t)'][t] for t, cf in enumerate(cash_flows)]
    df['NCF Tích lũy'] = cumulative_cf
    df['DCF Tích lũy'] = cumulative_discounted_cf

    # Lưu kết quả tính toán
    metrics = {
        "NPV": npv_value,
        "IRR": irr_value,
        "PP": pp,
        "DPP": dpp,
        "WACC": WACC,
        "Lifespan": N
    }

    return df, metrics

# --- HÀM 3: YÊU CẦU AI PHÂN TÍCH CHỈ SỐ ---
def get_ai_evaluation(data_for_ai, metrics, api_key):
    """Gửi các chỉ số và dòng tiền đến Gemini để nhận đánh giá chuyên sâu."""
    
    # Chuẩn bị dữ liệu để gửi cho AI
    metrics_str = "\n".join([f"{k}: {v:.2f}" if isinstance(v, (int, float)) and not math.isinf(v) and not math.isnan(v) else f"{k}: {v}" for k, v in metrics.items()])
    
    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính đầu tư. Hãy phân tích và đánh giá tính khả thi của dự án kinh doanh này dựa trên các chỉ số hiệu quả sau:

    ---
    THÔNG SỐ DỰ ÁN:
    {data_for_ai}
    
    ---
    CÁC CHỈ SỐ ĐÁNH GIÁ:
    {metrics_str} (WACC: {metrics['WACC'] * 100:.2f}%, Dòng đời dự án: {metrics['Lifespan']} năm)
    
    ---
    YÊU CẦU PHÂN TÍCH:
    1. NPV: Đánh giá dự án có nên được chấp nhận hay không (NPV > 0?).
    2. IRR so với WACC: Dự án có hiệu quả hơn chi phí vốn không (IRR > WACC?).
    3. PP & DPP: Nhận xét về tốc độ thu hồi vốn. Dự án có rủi ro về thanh khoản không?
    4. KẾT LUẬN: Đưa ra kết luận tổng thể và khuyến nghị rõ ràng (Nên thực hiện/Không nên thực hiện).
    
    Hãy trình bày dưới dạng một bài phân tích chuyên nghiệp, súc tích (khoảng 3-4 đoạn).
    """
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- CẤU TRÚC ỨNG DỤNG STREAMLIT ---
api_key = st.secrets.get("GEMINI_API_KEY")

with st.sidebar:
    st.subheader("Hướng Dẫn")
    st.info("""
    1. **Dán Nội dung:** Sao chép và dán toàn bộ nội dung phương án kinh doanh từ file Word vào ô bên dưới.
    2. **Lọc Dữ liệu:** Nhấn nút để AI trích xuất 6 thông số tài chính.
    3. **Phân tích:** Xem bảng dòng tiền, các chỉ số NPV, IRR, PP, DPP và nhận phân tích chuyên sâu từ AI.
    """)
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'.")

st.subheader("1. Nhập liệu Phương án Kinh doanh")
text_content = st.text_area(
    "Dán nội dung Phương án Kinh doanh (từ file Word) vào đây:", 
    height=300,
    key="plan_content",
    placeholder="Ví dụ: 'Dự án này cần 500 triệu VNĐ vốn đầu tư ban đầu, kéo dài 5 năm. Doanh thu 200 triệu/năm, Chi phí hoạt động 80 triệu/năm. WACC là 10% và Thuế TNDN là 20%.'"
)

if st.button("🚀 Lọc Dữ liệu Tài chính bằng AI"):
    if not api_key:
        st.error("Vui lòng cấu hình Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets.")
    elif not text_content:
        st.warning("Vui lòng dán nội dung phương án kinh doanh vào ô nhập liệu.")
    else:
        with st.spinner('Đang gửi dữ liệu đến AI để trích xuất thông số...'):
            extracted_data = extract_data_with_ai(text_content, api_key)
            st.session_state.extracted_data = extracted_data
            st.session_state.cash_flow_df = None
            st.session_state.metrics = None
        
        if st.session_state.extracted_data:
            # Sau khi trích xuất thành công, tính toán luôn
            df, metrics = calculate_metrics(st.session_state.extracted_data)
            st.session_state.cash_flow_df = df
            st.session_state.metrics = metrics
            st.success("Trích xuất và Tính toán thành công!")
            st.rerun() # Chạy lại để hiển thị kết quả

# --- HIỂN THỊ KẾT QUẢ ---

if st.session_state.extracted_data:
    st.subheader("✅ Thông số Dự án đã Trích xuất")
    
    data = st.session_state.extracted_data
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Vốn đầu tư (I₀)", f"{data['initial_investment']:,.0f} Triệu VNĐ")
    col2.metric("Dòng đời Dự án (N)", f"{data['project_lifespan']} Năm")
    col3.metric("WACC (r)", f"{data['wacc'] * 100:.2f}%")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Doanh thu Hàng năm", f"{data['annual_revenue']:,.0f} Triệu VNĐ")
    col5.metric("Chi phí Hàng năm", f"{data['annual_cost']:,.0f} Triệu VNĐ")
    col6.metric("Thuế suất", f"{data['tax_rate'] * 100:.0f}%")

# 2. Bảng Dòng tiền
if st.session_state.cash_flow_df is not None:
    st.subheader("2. Bảng Dòng tiền Dự án")
    st.dataframe(
        st.session_state.cash_flow_df.style.format({
            'Dòng tiền thuần (NCF)': '{:,.0f}',
            'Chiết khấu (1/(1+WACC)^t)': '{:.4f}',
            'Dòng tiền chiết khấu (DCF)': '{:,.0f}',
            'NCF Tích lũy': '{:,.0f}',
            'DCF Tích lũy': '{:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )

# 3. Chỉ số Đánh giá
if st.session_state.metrics is not None:
    st.subheader("3. Các Chỉ số Đánh giá Hiệu quả")
    metrics = st.session_state.metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Hàm hiển thị giá trị PP/DPP
    def format_period(value):
        if math.isinf(value) or math.isnan(value):
            return "Không Hoàn Vốn"
        return f"{value:.2f} Năm"
        
    col1.metric("NPV (Giá trị hiện tại ròng)", f"{metrics['NPV']:,.0f} Triệu VNĐ", delta_color="normal")
    col2.metric("IRR (Tỷ suất sinh lời nội bộ)", f"{metrics['IRR'] * 100:.2f}%" if not math.isnan(metrics['IRR']) else "N/A")
    col3.metric("PP (Thời gian hoàn vốn)", format_period(metrics['PP']))
    col4.metric("DPP (Hoàn vốn chiết khấu)", format_period(metrics['DPP']))

    # 4. Yêu cầu AI Phân tích
    st.subheader("4. Yêu cầu AI Phân tích Chỉ số")
    if st.button("🤖 Yêu cầu AI Phân tích và Đánh giá Dự án"):
        with st.spinner('Đang gửi kết quả tính toán và chờ Gemini phân tích...'):
            ai_evaluation = get_ai_evaluation(
                text_content, 
                metrics, 
                api_key
            )
            st.markdown("---")
            st.markdown("**Kết quả Phân tích & Khuyến nghị từ Gemini AI:**")
            st.info(ai_evaluation)

st.markdown("---")
st.caption("Lưu ý: Ứng dụng giả định Doanh thu và Chi phí không đổi qua các năm và bỏ qua Khấu hao, Vốn lưu động thay đổi và Giá trị thanh lý.")
