import streamlit as st
from setup import setup  # This should only set up configurations, not load data.
from query_manager import QueryManager
from feedback_manager import FeedbackManager
from trulens_eval import Tru

def initialize_tru():
    if 'tru' not in st.session_state:
        st.session_state.tru = Tru()

def main():
    st.title('🔍 Query and Feedback System')
    st.markdown("""
    Welcome to the **Query and Feedback System**! 🌟 This tool is designed to process your queries with precision and provide real-time feedback. 
    Use this system to harness powerful insights from your data, interactively and efficiently.
    """)

    initialize_tru()

    with st.sidebar:
        st.header('⚙️ Configuration')
        st.caption("Adjust system settings and initialize resources as needed.")
        if 'vector_index' not in st.session_state:
            if st.button('Initialize System', key='init_system'):
                with st.spinner('🔄 Setting up resources...'):
                    st.session_state.vector_index = setup()
                st.success('System initialized successfully! 🎉')

    st.header('📝 Submit Your Query')
    query = st.text_input('Enter your query here:', key='query_input')

    if st.button('Submit', key='submit_query'):
        if 'vector_index' not in st.session_state:
            with st.spinner('🔄 Initializing resources...'):
                st.session_state.vector_index = setup()

        query_manager = QueryManager(st.session_state.vector_index)
        st.session_state.response = query_manager.perform_query(query)

        feedback_manager = FeedbackManager(query_manager.query_engine)
        st.session_state.records = feedback_manager.record_query(query)

        # Custom HTML styling for response display
        st.markdown(f"**Response:** <div style='background-color:yellow;padding:10px;border-radius:5px;'>{st.session_state.response.response}</div>", unsafe_allow_html=True)
        st.write('Response:', st.session_state.response)
        st.write('Feedback Records:', st.session_state.records)

    manage_dashboard()

def manage_dashboard():
    st.header('🎮 Dashboard Management')
    port = 7000
    ip_address = "192.0.0.2"

    if st.button('🚀 Launch TRU Dashboard', key='launch_dashboard'):
        try:
            st.session_state.dashboard_process = st.session_state.tru.run_dashboard(port=port, force=True)
            st.success(f"🌐 Dashboard is now running on [http://{ip_address}:{port}](http://{ip_address}:{port})")
        except Exception as e:
            st.error(f"🚨 Error launching dashboard: {str(e)}")

    if st.button('🛑 Stop TRU Dashboard', key='stop_dashboard'):
        if 'dashboard_process' in st.session_state and st.session_state.dashboard_process is not None:
            st.session_state.dashboard_process.terminate()
            st.success("Dashboard has been stopped. 🛑")

if __name__ == '__main__':
    main()
