# ui/components/cards.py
import streamlit as st

def pillar_card(icon, title, subtitle, description):
    st.markdown(f"""
        <div style='border:1px solid #e5e7eb;border-top:4px solid #213b7d;
            border-radius:10px;padding:24px;background:#ffffff;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
          <div style='font-size:2rem;'>{icon}</div>
          <div style='font-family:Inter,sans-serif;font-size:0.9rem;font-weight:700;
              color:#171d1a;margin:10px 0 4px;'>{title}</div>
          <div style='font-family:Inter,sans-serif;font-size:0.7rem;font-weight:600;
              color:#6B7280;text-transform:uppercase;letter-spacing:0.05em;'>{subtitle}</div>
          <div style='font-family:Inter,sans-serif;font-size:0.8rem;color:#6B7280;
              margin-top:10px;'>{description}</div>
        </div>
    """, unsafe_allow_html=True)

def feature_card(icon, title, desc):
    st.markdown(f"""
        <div style='border:1px solid #e5e7eb;border-radius:8px;padding:16px;
            background:#fafafa;'>
          <div style='font-size:1.6rem;'>{icon}</div>
          <div style='font-family:Inter,sans-serif;font-size:1rem;font-weight:700;
              color:#171d1a;margin:6px 0 2px;'>{title}</div>
          <div style='font-family:Inter,sans-serif;font-size:0.8rem;color:#6B7280;'>
            {desc}
          </div>
        </div>
    """, unsafe_allow_html=True)