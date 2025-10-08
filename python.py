st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án")
        
        if wacc > 0:
            try:
                npv, irr, pp, dpp = calculate_project_metrics(df_cashflow, initial_investment, wacc)
                
                metrics_data = {
                    'NPV': npv,
                    'IRR': irr if not np.isnan(irr) else 0, # Dùng 0 để tránh lỗi format
                    'PP': pp,
                    'DPP': dpp
                }
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("NPV (Giá trị hiện tại thuần)", f"{npv:,.0f} VNĐ", delta=("Dự án có lời" if npv > 0 else "Dự án lỗ"))
                col2.metric("IRR (Tỷ suất sinh lời nội tại)", f"{irr:.2%}" if not np.isnan(irr) else "Không tính được")
                col3.metric("PP (Thời gian hoàn vốn)", f"{pp:.2f} năm" if isinstance(pp, float) or isinstance(pp, np.float64) else pp)
                col4.metric("DPP (Hoàn vốn có chiết khấu)", f"{dpp:.2f} năm" if isinstance(dpp, float) or isinstance(dpp, np.float64) else dpp)

                # ****************** Phân tích AI (Yêu cầu 4) ******************
                st.markdown("---")
                st.subheader("5. Phân tích Hiệu quả Dự án (AI)")
                
                if st.button("Yêu cầu AI Phân tích Chỉ số 🧠"):
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_evaluation(metrics_data, wacc, api_key)
                            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                            st.info(ai_result)
                    else:
                         st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng kiểm tra cấu hình Secrets.")

            except Exception as e:
                st.error(f"Có lỗi xảy ra khi tính toán chỉ số: {e}. Vui lòng kiểm tra các thông số đầu vào.")
        else:
            st.warning("WACC (Tỷ lệ chiết khấu) phải lớn hơn 0 để tính toán NPV và DPP.")

    else:
        st.warning("Vui lòng đảm bảo Dòng đời Dự án và Vốn Đầu tư đã được trích xuất thành công và có giá trị lớn hơn 0.")

else:
    st.info("Vui lòng tải lên file Word và nhấn nút 'Trích xuất Dữ liệu Tài chính bằng AI' để bắt đầu.")
